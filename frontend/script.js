document.addEventListener('DOMContentLoaded', function() {
    const imageUpload = document.getElementById('imageUpload');
    const imagePreview = document.getElementById('imagePreview');
    const previewContainer = document.getElementById('preview');
    const classifyBtn = document.getElementById('classifyBtn');
    const spinner = document.getElementById('spinner');
    const resultSection = document.getElementById('result');

    const API_URL = 'http://127.0.0.1:5000/predict';

    // Class descriptions
    const classDescriptions = [
        "Bad Infrastructure (Type A)",
        "Bad Infrastructure (Type B)",
        "Good Infrastructure (Type A)",
        "Good Infrastructure (Type B)"
    ];

    imageUpload.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.src = e.target.result;
                previewContainer.hidden = false;
                classifyBtn.disabled = false;
            }
            reader.readAsDataURL(file);
        }
    });

    classifyBtn.addEventListener('click', async function() {
        try {
            const formData = new FormData();
            formData.append('file', imageUpload.files[0]);

            spinner.hidden = false;
            resultSection.hidden = true;
            classifyBtn.disabled = true;

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (response.ok) {
                const qualityColor = data.is_good === 1 ? '#4CAF50' : '#f44336';
                
                let resultsHTML = `
                    <h2>Infrastructure Analysis</h2>
                    <div class="overall-quality" style="background-color: ${qualityColor}; color: white; padding: 15px; border-radius: 8px; margin: 15px 0;">
                        <h3>Overall Quality: ${data.is_good === 1 ? 'GOOD' : 'POOR'} Infrastructure</h3>
                        <p>Confidence: ${(data.quality_confidence * 100).toFixed(2)}%</p>
                    </div>
                    
                    <div class="quality-distribution">
                        <h3>Quality Distribution:</h3>
                        <div class="quality-bars">
                            <div class="quality-bar">
                                <span>Good Infrastructure: ${(data.good_infrastructure_prob * 100).toFixed(2)}%</span>
                                <div class="bar good" style="width: ${data.good_infrastructure_prob * 100}%"></div>
                            </div>
                            <div class="quality-bar">
                                <span>Poor Infrastructure: ${(data.bad_infrastructure_prob * 100).toFixed(2)}%</span>
                                <div class="bar bad" style="width: ${data.bad_infrastructure_prob * 100}%"></div>
                            </div>
                        </div>
                    </div>

                    <div class="specific-class">
                        <h3>Detailed Classification:</h3>
                        <p>Specific Class: ${classDescriptions[data.specific_class]}</p>
                        <p>Class Confidence: ${(data.class_confidence * 100).toFixed(2)}%</p>
                    </div>

                    <div class="individual-probabilities">
                        <h3>Individual Class Probabilities:</h3>
                        <div class="confidence-bars">
                `;
                
                data.individual_probs.forEach((prob, idx) => {
                    resultsHTML += `
                        <div class="confidence-bar">
                            <span>${classDescriptions[idx]}: ${(prob * 100).toFixed(2)}%</span>
                            <div class="bar ${idx < 2 ? 'bad' : 'good'}" style="width: ${prob * 100}%"></div>
                        </div>
                    `;
                });

                resultsHTML += `</div></div>`;
                resultSection.innerHTML = resultsHTML;
                resultSection.hidden = false;
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Error: ' + error.message);
        } finally {
            spinner.hidden = true;
            classifyBtn.disabled = false;
        }
    });
});