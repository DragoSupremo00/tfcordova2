//Properties for the webcam
let webProps = {
    facingMode: "environment",
    resizeWidth: 224,
    resizeHeight: 224,
    centerCrop: false
}
const webcamElement = document.getElementById('webcam');

//App execution
async function app()
{
    console.log("loading");
    model = await loadModel();
    const webcam = await tf.data.webcam(webcamElement, webProps);
    //Makes a new prediction when ready
    while (true) {
        tf.engine().startScope();
        let img = await webcam.capture();
        img = img.expandDims(0);
        result = await model.predict(img);
        img.dispose();
        let time1 = performance.now();
        await tf.nextFrame();
        let time2 = performance.now();
        let predictionTime = time2 - time1;
        console.log(tf.memory());
        console.log("Prediction time: " + predictionTime + "ms");
        tf.engine().endScope();
    }

}
async function loadModel() {
	try {
	    tfModelCache = await tf.loadLayersModel("https://raw.githubusercontent.com/DragoSupremo00/tfcordova2/master/model2/model.json");
	    return tfModelCache
	} catch (err) {
	  console.log(err)
	}
}

app();