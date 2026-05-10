document.addEventListener('DOMContentLoaded', () => {
    const batterySlider = document.getElementById('battery');
    const signalSlider = document.getElementById('signal');
    const storageSlider = document.getElementById('storage');
    const highQueueSlider = document.getElementById('high-queue');
    const lowQueueSlider = document.getElementById('low-queue');
    const healthSlider = document.getElementById('health');
    const commWindowToggle = document.getElementById('comm-window');
    
    const batteryVal = document.getElementById('battery-val');
    const signalVal = document.getElementById('signal-val');
    const storageVal = document.getElementById('storage-val');
    const highQueueVal = document.getElementById('high-queue-val');
    const lowQueueVal = document.getElementById('low-queue-val');
    const healthVal = document.getElementById('health-val');
    const windowStatus = document.getElementById('window-status');
    
    const decisionDisplay = document.getElementById('ai-decision');
    const autopilotBtn = document.getElementById('autopilot-btn');
    
    let autoPilotInterval = null;

    // Update labels and trigger API call
    function updateLabels() {
        batteryVal.textContent = batterySlider.value + '%';
        signalVal.textContent = signalSlider.value;
        storageVal.textContent = storageSlider.value + '%';
        highQueueVal.textContent = highQueueSlider.value;
        lowQueueVal.textContent = lowQueueSlider.value;
        healthVal.textContent = healthSlider.value + '%';
        
        if (commWindowToggle.checked) {
            windowStatus.textContent = 'OPEN';
            windowStatus.className = 'status-open';
        } else {
            windowStatus.textContent = 'CLOSED';
            windowStatus.className = 'status-closed';
        }
    }

    async function fetchAiDecision() {
        decisionDisplay.textContent = "ANALYZING...";
        decisionDisplay.className = "action-text waiting";

        const state = {
            battery_level: parseFloat(batterySlider.value),
            signal_strength: parseFloat(signalSlider.value),
            storage_usage: parseFloat(storageSlider.value),
            high_priority_queue: parseInt(highQueueSlider.value),
            low_priority_queue: parseInt(lowQueueSlider.value),
            communication_window: commWindowToggle.checked ? 1 : 0,
            system_health: parseFloat(healthSlider.value)
        };

        try {
            const response = await fetch('/schedule', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state)
            });
            const data = await response.json();
            
            if (data.recommended_action) {
                const action = data.recommended_action;
                let displayTxt = "";
                let cssClass = "action-text ";
                
                if (action === "transmit_high_priority") {
                    displayTxt = "🚀 TRANSMIT HIGH PRIORITY";
                    cssClass += "transmit";
                } else if (action === "transmit_low_priority") {
                    displayTxt = "📡 TRANSMIT LOW PRIORITY";
                    cssClass += "transmit";
                } else if (action === "conserve_power") {
                    displayTxt = "🔋 CONSERVE POWER";
                    cssClass += "conserve";
                } else if (action === "enter_low_power_mode") {
                    displayTxt = "💤 ENTER LOW POWER MODE";
                    cssClass += "sleep";
                } else {
                    displayTxt = "UNKNOWN ACTION";
                }

                decisionDisplay.textContent = displayTxt;
                decisionDisplay.className = cssClass;
            }
        } catch (err) {
            console.error(err);
            decisionDisplay.textContent = "CONNECTION ERROR";
            decisionDisplay.className = "action-text sleep";
        }
    }

    function handleChange() {
        updateLabels();
        fetchAiDecision();
    }

    // Attach listeners
    [batterySlider, signalSlider, storageSlider, highQueueSlider, lowQueueSlider, healthSlider].forEach(slider => {
        slider.addEventListener('input', handleChange);
    });
    commWindowToggle.addEventListener('change', handleChange);

    // Initial Fetch
    handleChange();

    // Auto Pilot Logic
    function randomJitter(value, min, max, step) {
        let change = (Math.random() - 0.5) * step * 5;
        let newVal = parseFloat(value) + change;
        return Math.min(Math.max(newVal, min), max);
    }

    autopilotBtn.addEventListener('click', () => {
        if (autoPilotInterval) {
            // Turn off
            clearInterval(autoPilotInterval);
            autoPilotInterval = null;
            autopilotBtn.textContent = "ENGAGE AUTO-PILOT";
            autopilotBtn.classList.remove('active');
        } else {
            // Turn on
            autopilotBtn.textContent = "AUTO-PILOT ACTIVE (CLICK TO STOP)";
            autopilotBtn.classList.add('active');
            
            autoPilotInterval = setInterval(() => {
                batterySlider.value = randomJitter(batterySlider.value, 0, 100, 1).toFixed(0);
                signalSlider.value = randomJitter(signalSlider.value, 0, 1, 0.05).toFixed(2);
                storageSlider.value = randomJitter(storageSlider.value, 0, 100, 1).toFixed(0);
                
                if (Math.random() > 0.8) {
                    highQueueSlider.value = Math.min(50, parseInt(highQueueSlider.value) + 1);
                }
                if (Math.random() > 0.5) {
                    lowQueueSlider.value = Math.min(50, parseInt(lowQueueSlider.value) + 1);
                }
                
                // Randomly toggle window 10% chance
                if (Math.random() > 0.9) {
                    commWindowToggle.checked = !commWindowToggle.checked;
                }
                
                handleChange();
            }, 1000);
        }
    });
});
