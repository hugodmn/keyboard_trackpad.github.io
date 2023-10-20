let canvas = document.getElementById('canvas');
let ctx = canvas.getContext('2d');
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
let isDrawing = false;
let model;
const labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',  'u', 'v', 'w', 'x', 'y', 'z'];
let currentSentence = "";

// Charger le modèle lors du chargement de la page
async function loadModel() {
    model = new onnx.InferenceSession();
    await model.loadModel('/training/model/emnist/resnet.onnx');
}
loadModel();

canvas.addEventListener('mousedown', (e) => { 
    isDrawing = true;
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
});


canvas.addEventListener('mouseup', () => { 
    isDrawing = false; 
});

canvas.addEventListener('mousemove', draw);

function updateSentenceDisplay() {
    document.getElementById('sentence').innerText = currentSentence;
}

// function draw(event) {
//     if (!isDrawing) return;

//     ctx.lineWidth = 10;
//     ctx.lineCap = 'round';
//     ctx.strokeStyle = 'white';

//     ctx.lineTo(event.clientX - canvas.offsetLeft, event.clientY - canvas.offsetTop);
//     ctx.stroke();
// }

function draw(event) {
    if (!isDrawing) return;

    // Valeur initiale
    ctx.lineWidth = 15;  // Augmenter la taille de la ligne pour un effet de brosse plus visible
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';

    const x = event.clientX - canvas.offsetLeft;
    const y = event.clientY - canvas.offsetTop;

    // Dessiner plusieurs fois avec différentes opacités pour obtenir l'effet de brosse
    const opacities = [1, 0.6, 0.3, 0.1];
    const offsets = [0, 3, 6, 9]; // Ces offsets déterminent à quelle distance du tracé central le pinceau sera appliqué

    for (let i = 0; i < opacities.length; i++) {
        ctx.globalAlpha = opacities[i];
        ctx.beginPath();
        ctx.moveTo(x - offsets[i], y - offsets[i]);
        ctx.lineTo(x + offsets[i], y + offsets[i]);
        ctx.stroke();
    }

    ctx.globalAlpha = 1; // Remettre l'opacité par défaut à 1 après le dessin
}


async function predictWithONNXModel() {
    // Redimensionnez l'image du canvas à 28x28
    let tmpCanvas = document.createElement('canvas');
    tmpCanvas.width = 28;
    tmpCanvas.height = 28;
    let tmpCtx = tmpCanvas.getContext('2d');

    tmpCtx.translate(28, 0);
    tmpCtx.rotate(Math.PI / 2);
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);


    // Extraire les pixels et les convertir en niveaux de gris
    let imgData = tmpCtx.getImageData(0, 0, 28, 28).data;

    let input = new Float32Array(28 * 28);
    for (let i = 0; i < imgData.length; i += 4) {
        let grayscale = (imgData[i]);// + imgData[i+1] + imgData[i+2]) / 3;
        // Normaliser entre -1 et 1
        input[i/4] = ((grayscale / 255) - 0.3309) / 0.17222;
    }


    // Prédire en utilisant le modèle ONNX
    console.log(input);
    let tensorInput = new onnx.Tensor(input, 'float32', [1, 1, 28, 28]);
    //console.log(tensorInput);
    let outputMap = await model.run([tensorInput]);
    let outputData = outputMap.values().next().value.data;
    //console.log(outputData);
    // Retourner la classe avec la plus haute probabilité
    return outputData.indexOf(Math.max(...outputData));
}


document.getElementById('predictButton').addEventListener('click', async function() {
    let predictionIndex = await predictWithONNXModel();
    let predictedLabel = labels[predictionIndex];
    currentSentence += predictedLabel;
    updateSentenceDisplay();

    // Clear the canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
});

document.getElementById('deleteButton').addEventListener('click', function() {
    // Remove the last character
    currentSentence = currentSentence.slice(0, -1);
    updateSentenceDisplay();
});

document.getElementById('spaceButton').addEventListener('click', function() {
    // Add a space
    currentSentence += " ";
    updateSentenceDisplay();
});

