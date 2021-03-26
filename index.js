let mobilenet;
let model;

const webcam = new Webcam(document.getElementById('wc'));

async function init(){
    await webcam.setup()
}

init();