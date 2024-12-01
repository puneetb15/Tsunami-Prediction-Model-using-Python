document.getElementById("predictionForm").addEventListener("submit", async function (event) {
    event.preventDefault();
  
    // Get form data
    const latitude = document.getElementById("latitude").value;
    const longitude = document.getElementById("longitude").value;
    const eq_magnitude = document.getElementById("eq_magnitude").value;
    const eq_depth = document.getElementById("eq_depth").value;
  
    const resultDiv = document.getElementById("result");
  
    try {
      // Send data to backend
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ latitude, longitude, eq_magnitude, eq_depth }),
      });
  
      if (!response.ok) throw new Error("Prediction failed!");
  
      const data = await response.json();
      resultDiv.innerHTML = `<h3>Tsunami Prediction: ${data.tsunami_prediction}</h3>
        <p><b>Details:</b><br>
        Latitude: ${data.details.latitude}<br>
        Longitude: ${data.details.longitude}<br>
        Magnitude: ${data.details.eq_magnitude}<br>
        Depth: ${data.details.eq_depth} km</p>`;
    } catch (error) {
      resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
  });
  