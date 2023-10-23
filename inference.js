let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
let isDrawing = false;
let model;
const labels = ['N/A', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',  'u', 'v', 'w', 'x', 'y', 'z'];

const sentences = [
    "yliess is a good teacher",
    "ai is amazing",
    "i love machine learning",
    "resnet is a great model",
    "pytorch is the best framework",
    "i love to learn new things",
    // ... add more sentences if you like
];

let currentSentence = "";
let currentCharIndex = 0;
setRandomSentence();


// Charger le modèle lors du chargement de la page
async function loadModel() {
    model = new onnx.InferenceSession();
    await model.loadModel('./training/model/emnist/only_letters/resnet.onnx');
    document.getElementById('predictButton').disabled = false;
}

loadModel();

// for the mouse
canvas.addEventListener('mousedown', (e) => { 
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
});


canvas.addEventListener('mouseup', () => { 
    isDrawing = false; 
});

canvas.addEventListener('mousemove', draw);
 // for phone users 

canvas.addEventListener('touchstart', function(e) {
    e.preventDefault();
    isDrawing = true;
    let touch = e.touches[0];
    ctx.beginPath();
    ctx.moveTo(touch.clientX - canvas.offsetLeft, touch.clientY - canvas.offsetTop);
});

canvas.addEventListener('touchend', function() {
    e.preventDefault();
    isDrawing = false;
});

canvas.addEventListener('touchmove', function(e) {
    e.preventDefault();
    if (!isDrawing) return;
    let touch = e.touches[0];
    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.lineTo(touch.clientX - canvas.offsetLeft, touch.clientY - canvas.offsetTop);
    ctx.stroke();
});




function updateSentenceDisplay() {
    document.getElementById('sentence').innerText = currentSentence;
}

function draw(event) {
    if (!isDrawing) return;

    ctx.lineWidth = 10;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
    ctx.stroke();
}




async function predictWithONNXModel() {
    // Redimensionnez l'image du canvas à 28x28
    let tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    let tmpCtx = tmpCanvas.getContext('2d');

    tmpCtx.drawImage(canvas, 0, 0, 28, 28);


    // Extraire les pixels et les convertir en niveaux de gris
    let NOT_ZERO_NUMB = 0
    let pix_num = 0
    let imgData = tmpCtx.getImageData(0, 0, 28, 28).data;


    for (let i = 0; i < imgData.length; i+=1){
        if (imgData[i]!=0){
            NOT_ZERO_NUMB += 1; }
        pix_num += 1
    }

    let input = new Float32Array(28 * 28);
    //let array2D = [];
    for (let i = 0; i < imgData.length; i += 4) {
        let grayscale = 0
        if (imgData[i]!=0)  {grayscale = 255}
        if (imgData[i+1]!=0)  {grayscale = 255}
        if (imgData[i+2]!=0)  {grayscale = 255}
        grayscale = (((grayscale/ 255)) - 0.1736) / 0.3317;
        // Normaliser entre -1 et 1
        input[i/4] = grayscale
    }

    let transposedFlattened = new Float32Array(28 * 28);
    // Flattening 
    for (let y = 0; y < 28; y++) {
        for (let x = 0; x < 28; x++) {
            let originalIndex = y * 28 + x;
            let transposedIndex = x * 28 + y;  // Here's where the transpose happens
            transposedFlattened[transposedIndex] = input[originalIndex];
        }
    }
    input = transposedFlattened;

    // let processedCanvas = document.getElementById('processedCanvas');
    // let processedCtx = processedCanvas.getContext('2d');
    // processedCtx.drawImage(tmpCanvas, 0, 0, 28, 28);

    let tensorInput = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);
    let outputMap = await model.run([tensorInput]);
    
    let outputData = outputMap.values().next().value.data;

    //console.log(outputData);
    // Retourner la classe avec la plus haute probabilité
    console.log(outputData.indexOf(Math.max(...outputData)))
    return outputData.indexOf(Math.max(...outputData));
}

function setNextCharIndex() {
    currentCharIndex++;
    while (currentSentence[currentCharIndex] === ' ') {
        currentCharIndex++;
    }
}


document.getElementById('predictButton').addEventListener('click', async function() {
    let predictionIndex = await predictWithONNXModel();
    let predictedLabel = labels[predictionIndex];
    
    document.getElementById('prediction').textContent = `Prediction: ${predictedLabel}`;
    
    // Check if prediction matches the current character in the sentence

    if (predictedLabel === currentSentence[currentCharIndex]) {
        // currentCharIndex++;
        setNextCharIndex();
        if (currentCharIndex >= currentSentence.length) { // If we've reached the end of the sentence
            alert("Well done! Starting a new sentence.");
            setRandomSentence();
        }
    } else {
        // Display current character in red if it's wrong
        let sentenceElem = document.getElementById('sentence');
        sentenceElem.childNodes[currentCharIndex].style.color = 'red';
    }

    // Update the display accordingly
    updateSentenceDisplay();

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

});

function setRandomSentence() {
    // currentSentence = sentences[Math.floor(Math.random() * sentences.length)];
    // currentCharIndex = 0;
    // updateSentenceDisplay();
    let randomIndex = Math.floor(Math.random() * sentences.length);
    currentSentence = sentences[randomIndex];
    document.getElementById('sentence').innerText = currentSentence;
    // move to the first non-space character
    while (currentSentence[currentCharIndex] === ' ') {
        currentCharIndex++;
    }
    updateSentenceDisplay();
}


function updateSentenceDisplay() {
    let displayedSentence = [...currentSentence].map((char, index) => {
        if (index < currentCharIndex) {
            return `<span style="color:green">${char}</span>`;
        } else if (index === currentCharIndex) {
            return `<span style="color:orange">${char}</span>`;
        }
        return char;
    }).join('');
    
    document.getElementById('sentence').innerHTML = displayedSentence;
}
