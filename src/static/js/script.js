 const valueRez = document.getElementById('tmpRez') 
 const resultText = document.getElementById('full-text')
 
 // Функция для очистки холста
 function clearCanvas() {
    fetch('/clear_canvas', { method: 'POST' })
        .then(response => console.log('Canvas cleared'))
        .catch(error => console.error('Error:', error));
}

// Слушаем нажатие клавиш
window.addEventListener('keydown', function(event) {
    if (event.key === 'c' || event.key === 'C') {  // Если нажата клавиша C
        clearCanvas();
    }else if (event.key === 's' || event.key === 'S') {  // Если нажата клавиша S
        resultText.innerHTML += valueRez.innerHTML
        // fetch('/get_letter', { method: 'POST' })
        // .then(response => console.log('Letter ok'))
        // .catch(error => console.error('Error:', error));
    }
});

setInterval(() => {
    fetch('/get_letters', { method: 'POST' })
        .then(response => {
            console.log("Ответ сервера:", response); // Лог заголовков
            if (!response.ok) throw new Error(`Ошибка сервера: ${response.status}`);
            return response.json(); // Преобразуем JSON
        })
        .then(data => {
            console.log("Полученные данные:", data);
            valueRez.innerText = data.letter;
        })
        .catch(error => console.error('Ошибка:', error));
}, 5000);


document.getElementById('save-letter').addEventListener('click',()=>{
    resultText.innerHTML += valueRez.innerHTML
})
document.getElementById('remove-letter').addEventListener('click',()=>{
    resultText.innerHTML = resultText.innerHTML.slice(0, -1)
})
document.getElementById('space').addEventListener('click',()=>{
    resultText.innerHTML += ' '
})
document.getElementById('clear-screen').addEventListener('click',()=>{
    clearCanvas()
})
document.getElementById('start-drawing').addEventListener('click',()=>{
    fetch('/set_is_write', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ is_write: true }) // Можно отправить любое значение, если нужно
    })
    .then(response => {
        if (response.ok) {
            console.log("Запрос успешно отправлен");
        } else {
            console.error("Ошибка при отправке запроса");
        }
    })
    .catch(error => console.error("Ошибка сети:", error));
})
document.getElementById('stop-drawing').addEventListener('click',()=>{
    fetch('/set_is_write', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ is_write: false }) // Можно отправить любое значение, если нужно
    })
    .then(response => {
        if (response.ok) {
            console.log("Запрос успешно отправлен");
        } else {
            console.error("Ошибка при отправке запроса");
        }
    })
    .catch(error => console.error("Ошибка сети:", error));
})
