const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

// Mapping from model's output index to corresponding character
const classMapping = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't'];

let modelSession = null;

// Load the model during the page load
window.onload = async function() {
    modelSession = new onnx.InferenceSession();
    await modelSession.loadModel('path_to_your_model.onnx');
};

canvas.addEventListener('mousedown', () => { drawing = true; });
canvas.addEventListener('mouseup', () => { drawing = false; ctx.beginPath(); });
canvas.addEventListener('mousemove', draw);

function draw(event) {
    if (!drawing) return;
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function predict() {
    if(!modelSession) {
        console.error("Model not loaded yet!");
        return;
    }

    const inputData = preprocessCanvasData(canvas);
    const inputTensor = new onnx.Tensor(inputData, 'float32', [1, 1, 28, 28]);
    const outputMap = await modelSession.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;

    const predictionIndex = outputData.indexOf(Math.max(...outputData));
    const predictedChar = classMapping[predictionIndex];
    
    document.getElementById('sentenceField').value += predictedChar;
    clearCanvas();
}

function addSpace() {
    document.getElementById('sentenceField').value += ' ';
}

function deleteCharacter() {
    const currentSentence = document.getElementById('sentenceField').value;
    document.getElementById('sentenceField').value = currentSentence.slice(0, -1);
}

function preprocessCanvasData(canvas) {
    const tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;

    const tmpCtx = tmpCanvas.getContext('2d');
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tmpCtx.getImageData(0, 0, 28, 28);
    let data = Array.from(imageData.data);

    // Convert to grayscale
    const grayscaleData = [];
    for (let i = 0; i < data.length; i += 4) {
        const avg = (data[i] + data[i + 1] + data[i + 2]) / 3;
        grayscaleData.push(avg);
    }

    // Normalize using variance and std
    const mean = grayscaleData.reduce((sum, value) => sum + value, 0) / grayscaleData.length;
    const variance = grayscaleData.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / grayscaleData.length;
    const std = Math.sqrt(variance);

    const normalizedData = grayscaleData.map(value => (value - mean) / std);

    return normalizedData;
}
