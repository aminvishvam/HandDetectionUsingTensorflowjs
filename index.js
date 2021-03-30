let mobilenet;
let model;
const webcam = new Webcam(document.getElementById('wc'));
console.log(webcam)
const dataset = new handData();
var yesSemple = 0, noSamples = 0, callMeSamples = 0, peaceSamples = 0;
let isPredicting = false;

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
    const layer = mobilenet.getLayer('conv_pw_13_relu');
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

async function train() {
    const metrics = ['loss', 'val_loss', 'accuracy', 'val_accuracy'];
	const container = { name: 'Model Training', styles: { height: '640px' } };
	const fitCallbacks = tfvis.show.fitCallbacks(container, metrics);

    dataset.ys = null;
    dataset.encodeLabels(4);
    model = tf.sequential({
        layers: [
            tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
            tf.layers.dense({ units: 100, activation: 'relu' }),
            tf.layers.dense({ units: 4, activation: 'softmax' })
        ]
    });
    const optimizer = tf.train.adam(0.0001);
    model.compile({ optimizer: optimizer, loss: 'categoricalCrossentropy' });
    model.summary();
    let loss = 0;
    tfvis.show.modelSummary({name: 'Model Architecture'}, model)
    model.fit(dataset.xs, dataset.ys, {
        epochs: 15,
        callbacks: fitCallbacks
        // callbacks: {
        //     onBatchEnd: async (batch, logs) => {
        //         loss = logs.loss.toFixed(5);
        //         console.log('LOSS: ' + loss);
        //     }
        // }
    });
    
}


function handleButton(elem) {
    switch (elem.id) {
        case "0":
            yesSemple++;
            document.getElementById("yessamples").innerText = "Yes samples:" + yesSemple;
            break;
        case "1":
            noSamples++;
            document.getElementById("nosamples").innerText = "No samples:" + noSamples;
            break;
        case "2":
            callMeSamples++;
            document.getElementById("callmesamples").innerText = "Callme samples:" + callMeSamples;
            break;
        case "3":
            peaceSamples++;
            document.getElementById("peacesamples").innerText = "Peace samples:" + peaceSamples;
            break;
    }
    label = parseInt(elem.id);
    const img = webcam.capture();
    // console.log(typeof(img));
    dataset.addExample(mobilenet.predict(img), label);
}

async function predict() {
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const predictions = model.predict(activation);
            return predictions.as1D().argMax();
        });
        const classId = (await predictedClass.data())[0];
        var predictionText = "";
        switch (classId) {
            case 0:
                predictionText = "I see Yes Sign";
                break;
            case 1:
                predictionText = "I see No Sign";
                break;
            case 2:
                predictionText = "I see Callme Sign";
                break;
            case 3:
                predictionText = "I see Peace Sign";
                break;
        }
        document.getElementById("prediction").innerText = predictionText;
        predictedClass.dispose();
        await tf.nextFrame();
    }
}


function doTraining() {
    train();
}

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
    tf.tidy(() => mobilenet.predict(webcam.capture()));
    
}

init();