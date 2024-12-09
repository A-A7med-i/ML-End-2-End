document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    const formData = new FormData(e.target);
    const data = Object.fromEntries(formData.entries());

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        const result = await response.json();
        document.getElementById('result').innerHTML = `Prediction: ${result.prediction}`;
    } catch (error) {
        document.getElementById('result').innerHTML = `Error: ${error.message}`;
    }
});
