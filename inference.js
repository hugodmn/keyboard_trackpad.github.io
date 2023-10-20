document.addEventListener("DOMContentLoaded", function () {
    const canvas = document.getElementById("canvas");
    const ctx = canvas.getContext("2d");
    const spaceButton = document.getElementById("spaceButton");
    const deleteButton = document.getElementById("deleteButton");
    const recognizedText = document.getElementById("predictedText");

    let drawing = false;

    canvas.addEventListener("mousedown", () => {
        drawing = true;
        ctx.beginPath();
    });

    canvas.addEventListener("mousemove", draw);
    canvas.addEventListener("mouseup", () => {
        drawing = false;
    });

    function draw(e) {
        if (!drawing) return;
        ctx.lineWidth = 10;
        ctx.lineCap = "round";
        ctx.strokeStyle = "black";
        ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
        ctx.stroke();
    }

    const predictButton = document.getElementById("predictButton");
    predictButton.addEventListener("click", predictCharacter);

    function predictCharacter() {
        const canvasData = canvas.toDataURL(); // Get the drawn image data

        // Perform character recognition using the ONNX model
        const image = new Image();
        image.src = canvasData;

        image.onload = function () {
            // Resize the image to the model's input size (e.g., 28x28 pixels)
            const resizedCanvas = document.createElement("canvas");
            const resizedCtx = resizedCanvas.getContext("2d");
            resizedCanvas.width = 28;
            resizedCanvas.height = 28;
            resizedCtx.drawImage(image, 0, 0, 28, 28);

            // Preprocess the resized image
            const tensor = preprocessImage(resizedCanvas);

            // Perform inference using the ONNX model (replace 'model.onnx' with your model file)
            onnx.InferenceSession.create().then(function (onnxModel) {
                return onnxModel.loadModel('/training/model/emnist/best_model.onnx').then(function () {
                    const input = new onnx.Tensor(new Float32Array(tensor.data), 'float32');
                    return onnxModel.run([input]);
                }).then(function (output) {
                    // Display the recognized character
                    const recognizedCharacter = String.fromCharCode(65 + Math.round(output[0].data[0] * 25));
                    recognizedText.textContent += recognizedCharacter;
                }).then(function () {
                    // Clear the canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                });
            });
        };
    }

    spaceButton.addEventListener("click", function () {
        recognizedText.textContent += " ";
    });

    deleteButton.addEventListener("click", function () {
        const currentText = recognizedText.textContent;
        recognizedText.textContent = currentText.substring(0, currentText.length - 1);
    });

    // Preprocess the image (normalize and convert to a tensor)
    function preprocessImage(image) {
        const imageData = image.getContext("2d").getImageData(0, 0, image.width, image.height).data;
        const inputArray = new Float32Array(imageData.length);

        for (let i = 0; i < imageData.length; i++) {
            inputArray[i] = imageData[i] / 255; // Normalize pixel values
        }

        return new onnx.Tensor(inputArray, 'float32', [1, 28, 28, 1]);
    }
});
