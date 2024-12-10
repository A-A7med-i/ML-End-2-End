document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const formData = {
        f1: parseInt(document.getElementById('f1').value),
        f2: parseInt(document.getElementById('f2').value),
        f3: parseFloat(document.getElementById('f3').value),
        f4: parseFloat(document.getElementById('f4').value),
        f5: parseFloat(document.getElementById('f5').value),
        f6: parseInt(document.getElementById('f6').value),
        f7: parseFloat(document.getElementById('f7').value)
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });

        const responseData = await response.json();
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'block';
        resultDiv.innerHTML = `<h3>Prediction Result: ${responseData.prediction}</h3>`;
    } catch (error) {
        console.error('Error:', error);
        alert('Error making prediction');
    }
});