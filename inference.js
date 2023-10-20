const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
let drawing = false;

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
    const session = new onnx.InferenceSession();
    await session.loadModel('./training/model/emnist/best_model.onnx');

    const inputData = preprocessCanvasData(canvas);
    const inputTensor = new onnx.Tensor(inputData, 'float32', [1, 1, 28, 28]);
    const outputMap = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;

    const prediction = outputData.indexOf(Math.max(...outputData));
    
    // Convert the prediction to the appropriate letter/digit
    const predictedChar = convertToChar(prediction);
    
    // Add predicted char to the sentence field
    document.getElementById('sentenceField').value += predictedChar;
    
    // Clear canvas after prediction
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
    // Convert canvas to grayscale 28x28 and normalize
    // This function depends on your model's input requirements
    // Implement accordingly.
}

function convertToChar(predictionIndex) {
    // Map the prediction index to the corresponding letter/digit
    // Assuming the model predicts numbers first and then letters
    // Implement this function based on your model's classes
}

