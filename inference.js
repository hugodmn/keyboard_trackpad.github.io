

document.addEventListener('DOMContentLoaded', function () {

let canvas = document.getElementById('the-canvas');
let ctx = canvas.getContext('2d');
let isDrawing = false;
let sentenceElement = document.getElementById('sentence');
let currentSentence = '';

// Load the ONNX model
let session = new onnx.InferenceSession();
session.loadModel('./training/model/emnist/best_model.onnx');

// Function to handle the drawing in canvas
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

function startDrawing(e) {
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

function draw(e) {
    if (!isDrawing) return;
    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
}

function stopDrawing() {
    isDrawing = false;
}

// Function to predict what's drawn on the canvas
async function predict() {
    // Here, preprocess the canvas content (image) to fit your model input requirements.
    // This might include resizing the image, converting it to a tensor, normalizing it, etc.

    // For demonstration, assuming the preprocessed input is stored in 'inputTensor'.
    const inputTensor = {};  // Your actual preprocessing will replace this line

    // Running the model to get the prediction
    const outputMap = await session.run([inputTensor]);
    const outputData = outputMap.values().next().value.data;  // or another appropriate way to extract data

    // Convert model output to readable result (i.e., the predicted character)
    const predictedCharacter = '';  // Add your logic to extract character from 'outputData'

    // Add the predicted character to the sentence
    currentSentence += predictedCharacter;
    sentenceElement.innerText = currentSentence;

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function addSpace() {
    currentSentence += ' ';
    sentenceElement.innerText = currentSentence;
}

function deleteCharacter() {
    currentSentence = currentSentence.slice(0, -1);  // Remove the last character
    sentenceElement.innerText = currentSentence;
}

});