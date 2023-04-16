// JavaScript code to handle button clicks
document.getElementById("adjbtn").addEventListener("click", function() {
    sendRequest("adjectives");
});

document.getElementById("alphabtn").addEventListener("click", function() {
    sendRequest("alphabet");
});

document.getElementById("bodbtn").addEventListener("click", function() {
    sendRequest("body");
});

document.getElementById("clobtn").addEventListener("click", function() {
    sendRequest("clothes");
});

document.getElementById("colbtn").addEventListener("click", function() {
    sendRequest("clothes");
});

document.getElementById("dribtn").addEventListener("click", function() {
    sendRequest("drinks");
});

document.getElementById("emobtn").addEventListener("click", function() {
    sendRequest("emotions");
});

document.getElementById("fambtn").addEventListener("click", function() {
    sendRequest("family");
});

document.getElementById("foobtn").addEventListener("click", function() {
    sendRequest("food");
});

document.getElementById("fruibtn").addEventListener("click", function() {
    sendRequest("fruit");
});

document.getElementById("houbtn").addEventListener("click", function() {
    sendRequest("house");
});

document.getElementById("numbtn").addEventListener("click", function() {
    sendRequest("number");
});

document.getElementById("objbtn").addEventListener("click", function() {
    sendRequest("objects");
});

document.getElementById("placebtn").addEventListener("click", function() {
    sendRequest("place");
});

document.getElementById("probtn").addEventListener("click", function() {
    sendRequest("pronouns");
});

document.getElementById("shapesbtn").addEventListener("click", function() {
    sendRequest("shapes");
});

document.getElementById("vegbtn").addEventListener("click", function() {
    sendRequest("vegetables");
});

document.getElementById("verbbtn").addEventListener("click", function() {
    sendRequest("verb");
});

document.getElementById("weabtn").addEventListener("click", function() {
    sendRequest("weather");
});





// Function to send HTTP request to Flask server
function sendRequest(buttonInfo) {
    var xhttp = new XMLHttpRequest();
        xhttp.onreadystatechange = function() {
            if (this.readyState == 4 && this.status == 200) {
            console.log("Response from Flask server: " + this.responseText);
            }
    };
    xhttp.open("GET", "/button-clicked?info=" + buttonInfo, true);
    xhttp.send();
}