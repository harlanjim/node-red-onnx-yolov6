module.exports = function(RED) {
    function OnnxYoloDetect(config) {
        const jimp = require('jimp')
        const ort = require('onnxruntime-node')
        const fs = require('fs')
        const pureimage = require("pureimage")
        var jpeg = require('jpeg-js');

        RED.nodes.createNode(this,config)
        this.predLevel = config.predLevel || 0.70
        this.ouiLevel = config.ouiLevel || 0.30
        this.passthru = config.passthru || "false"
        this.lineColor = config.lineColor || "yellow"
        this.provider = config.provider || 'cpu'
        this.detectObjects = config.detectObjects || ["person"]
        var node = this
        var session = null 
        const InferenceSession = ort.InferenceSession;
        const tensor = ort.Tensor
        const width = 640
        const height = 480
        const class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
        'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
         'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
         'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 
         'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 
         'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
         'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        
        console.log(this.detectObjects)
        async function loadFont() {
            node.fnt = pureimage.registerFont(__dirname + '/SourceSansPro-Regular.ttf','Source Sans Pro');
            node.fnt.load();
        }
        loadFont();

        async function loadModel() {
            const sessionOption = { executionProviders: [node.provider] }
            session = await InferenceSession.create(__dirname + '/yolov6s.onnx', sessionOption)
            node.ready = true;
            node.status({fill:'green', shape:'dot', text:'Model ready'});
        }
        node.status({fill:'yellow', shape:'ring', text:'Loading model...'});
        loadModel();

        // Main Processing
        async function inference(msg) {
            const image = await jimp.read(msg.payload)
            //if (node.passthru === "bbox") { jimg = jimp.read(m.payload); }
            if (node.passthru === "true") { msg.image = msg.payload; }
            const dims = [1, 3, height, width]
            const imageTensor = await imageDataToTensor(image, dims)
            const feeds = {}
            feeds[session.inputNames[0]] = imageTensor
            const results = await session.run(feeds)
            const output = results[session.outputNames[0]]
            const keepers = await processOutput(output)   
            msg.predictions = keepers
            if (node.passthru === "bbox" && keepers.length > 0) {
                writeImageWithAnnotation(msg, image, keepers)
            }
            node.send(msg);
        }

        async function processOutput(output) {
            var predictions = new Array()
            for (let i = 0; i < output.data.length; i += output.dims[2]) {
                    var confValue = output.data[i + 4]
                    if (confValue > node.predLevel) {
                        // Build array of all the classes probabilities
                        var classes = new Array()
                        for (let y = 5; y < output.dims[2]; y++) {
                            classes.push(output.data[i+y])
                        }
                        // Then get the class with the highest probability
                        const classId = argmax(classes) 
                        // is it one we are looking for?
                        if (node.detectObjects.indexOf(class_names[classId]) > -1 && classes[classId] >= node.predLevel) {
                            predictions.push({
                                confValue: confValue,
                                classId: classId,
                                className: class_names[classId],
                                classProb: classes[classId],
                                boxes: {
                                    x: output.data[i + 0] - (output.data[i + 2] / 2),
                                    y: output.data[i + 1] - (output.data[i + 3] / 2),
                                    w: output.data[i + 2],
                                    h: output.data[i + 3]
                                }
                            })
                        }
                    }
            }
            // Sort predictions by probability
            predictions.sort((a, b) => {
                return b.classProb - a.classProb
            })
            // Check for overlap
            var keepIndices = [0]; // First one is a keeper since we sorted.
            for (var i = 1; i < predictions.length; i++) {
                if (isPredictionAKeeper(predictions, keepIndices, predictions[i].boxes)) {
                    keepIndices.push(i)
                }
            }
            var keepers = predictions.filter( (el, i) => keepIndices.some(j => i === j))
            //console.log(keepers);
            return keepers
        }

        function argmax(vector) {
            var index = 0
            var value = vector[0]
            for (let i = 1; i < vector.length; i++) {
                if (vector[i] > value) {
                    index = i
                    value = vector[i]
                }
            }
            return index
        }

        function isPredictionAKeeper(pred, keepIndices, box) {
            // Filter predictions by keeper indices
            var keepers = pred.filter( (el, i) => keepIndices.some(j => i === j))
            // The rectangle we are checking against keepers
            var xmin0 = box.x 
            var xmax0 = box.x + box.w
            var ymin0 = box.y
            var ymax0 = box.y + box.h
            for (var j = 0; j < keepers.length; j++) {
                var xmin1 = keepers[j].boxes.x 
                var xmax1 = keepers[j].boxes.x + keepers[j].boxes.w
                var ymin1 = keepers[j].boxes.y
                var ymax1 = keepers[j].boxes.y + keepers[j].boxes.h
                var si = Math.max(0, Math.min(xmax0, xmax1) - Math.max(xmin0,xmin1)) * Math.max(0, Math.min(ymax0, ymax1) - Math.max(ymin0, ymin1))
                var s0 = Math.max(0, xmax0 - xmin0) * Math.max(0, ymax0 - ymin0)
                var s1 = Math.max(0, xmax1 - xmin1) * Math.max(0, ymax1 - ymin1)
                var su = s0 + s1 - si
                var oui = si / su
                if (oui > node.ouiLevel) {
                    return false; // We intersect with a keeper so throw this away.
                }
            }
            return true;
        }

        function writeImageWithAnnotation(msg, image, predictions) {
            var pimg = pureimage.make(image.bitmap.width, image.bitmap.height);
            const ctx = pimg.getContext('2d')
            var scale = parseInt((image.bitmap.width+image.bitmap.height) / 500 + 0.5)
            ctx.bitmap.data = image.bitmap.data
            for (var i=0; i < predictions.length; i++) {
                var score = parseInt(predictions[i].classProb * 100)
                ctx.fillStyle = node.lineColor
                ctx.strokeStyle = node.lineColor
                ctx.font = scale * 8 + "pt 'Source Sans Pro'"
                ctx.fillText (predictions[i].className + " " + score + "%", predictions[i].boxes.x + 4, predictions[i].boxes.y - 4)
                ctx.lineWidth = 1
                ctx.lineJoin = 'bevel'
                ctx.strokeRect(predictions[i].boxes.x, predictions[i].boxes.y, predictions[i].boxes.w, predictions[i].boxes.h)
                ctx.strokeRect(predictions[i].boxes.x+1, predictions[i].boxes.y+1, predictions[i].boxes.w-2, predictions[i].boxes.h-2)
            }
            msg.image = jpeg.encode(pimg, 70).data
        }

        async function imageDataToTensor(image, dims) {
            var imageBufferData = image.bitmap.data
            const [redArray, greenArray, blueArray] = new Array(new Array(), new Array(), new Array())
            for (let i = 0; i < imageBufferData.length; i += 4) {
                redArray.push(imageBufferData[i])
                greenArray.push(imageBufferData[i + 1])
                blueArray.push(imageBufferData[i + 2])
                // skip data[i + 3] to filter out the alpha channel
            }
            const transposedData = redArray.concat(greenArray).concat(blueArray)
            let i, l = transposedData.length
            const float32Data = new Float32Array(dims[1] * dims[2] * dims[3])
            for (i = 0; i < l; i++) {
                float32Data[i] = transposedData[i] / 255 // convert to float
            }
            const inputTensor = new tensor("float32", float32Data, dims)
            return inputTensor;
        }

        node.on('input', function(msg) {
            try {
                if (node.ready) {
                    msg.image = msg.payload;
                    if (typeof msg.payload === "string") {
                        if (msg.payload.startsWith("http")) {
                            getImage(msg);
                            return;
                        }
                        else if (msg.payload.startsWith("data:image/jpeg")) {
                            msg.payload = Buffer.from(msg.payload.split(";base64,")[1], 'base64');
                        }
                        else { msg.payload = fs.readFileSync(msg.payload); }
                    }
                    inference(msg);
                }
            } catch (error) {
                node.error(error, msg);
            }        
        });
        node.on("close", function () {
            node.status({});
            node.ready = false;
            session.dispose();
            session = null;
            node.fnt = null;
        });
    }
    RED.nodes.registerType("onnx-yolo",OnnxYoloDetect);
}
