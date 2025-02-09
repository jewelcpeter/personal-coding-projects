document.addEventListener("DOMContentLoaded", function () {
    loadMedications();
    checkAlarms(); // Start checking for alarms
    setInterval(checkAlarms, 60000); // Check every minute
});

function addMedication() {
    const medName = document.getElementById("medication").value;
    const medTime = document.getElementById("time").value;

    if (medName === "" || medTime === "") {
        alert("Please enter both medication name and time.");
        return;
    }

    const medList = document.getElementById("medList");

    const li = document.createElement("li");
    li.innerHTML = `${medName} - ${medTime} <button onclick="markAsTaken(this)">Taken</button>`;

    medList.appendChild(li);

    saveMedication(medName, medTime);

    document.getElementById("medication").value = "";
    document.getElementById("time").value = "";
}

function markAsTaken(button) {
    button.parentElement.classList.toggle("completed");
}

function saveMedication(name, time) {
    let meds = JSON.parse(localStorage.getItem("medications")) || [];
    meds.push({ name, time });
    localStorage.setItem("medications", JSON.stringify(meds));
}

function loadMedications() {
    let meds = JSON.parse(localStorage.getItem("medications")) || [];
    const medList = document.getElementById("medList");

    meds.forEach(med => {
        const li = document.createElement("li");
        li.innerHTML = `${med.name} - ${med.time} <button onclick="markAsTaken(this)">Taken</button>`;
        medList.appendChild(li);
    });
}

// Function to check for medication alarms
function checkAlarms() {
    const currentTime = new Date();
    const currentHour = String(currentTime.getHours()).padStart(2, "0");
    const currentMinutes = String(currentTime.getMinutes()).padStart(2, "0");
    const formattedTime = `${currentHour}:${currentMinutes}`;

    let meds = JSON.parse(localStorage.getItem("medications")) || [];

    meds.forEach(med => {
        if (med.time === formattedTime) {
            alert(`Time to take your medication: ${med.name}`);
            playAlarmSound();
        }
    });
}

// Function to play an alarm sound
function playAlarmSound() {
    const alarm = new Audio("https://www.soundjay.com/button/beep-07.wav"); // Use an online sound
    alarm.play();
}
