 // Получаем элементы DOM
 const valueRez = document.getElementById('tmpRez')  // Элемент, который отображает результат распознавания
 const resultText = document.getElementById('full-text')  // Элемент, куда добавляется текст
  
 // Функция для очистки холста
 function clearCanvas() {
    // Отправляем запрос на очистку холста на сервер
    fetch('/clear_canvas', { method: 'POST' })
        .then(response => console.log('Canvas cleared')) // Успешное выполнение
        .catch(error => console.error('Error:', error)); // Ошибка при запросе
}
function statusDrow(value){
    fetch('/set_is_write', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ is_write: value }) // Можно отправить любое значение, если нужно
    })
    .then(response => {
        if (!response.ok){
            console.error("Ошибка при отправке запроса");
        }
    })
    .catch(error => console.error("Ошибка сети:", error));
}

// Слушаем нажатие клавиш
window.addEventListener('keydown', function(event) {
    if (event.key === 'c' || event.key === 'C') {  // Если нажата клавиша C
        clearCanvas(); // Очищаем холст
    }else if (event.key === 's' || event.key === 'S') {  // Если нажата клавиша S
        resultText.innerHTML += valueRez.innerHTML
    }else if (event.key === ' ' || event.key === ' ') {  // Если нажата клавиша S
        resultText.innerHTML += ' '
    }else if (event.key === 'd' || event.key === 'D') {  // Если нажата клавиша S
        resultText.innerHTML = resultText.innerHTML.slice(0, -1)
    }
});
// Получаем данные с сервера с интервалом 5 секунд (можно сократить или увеличить)
setInterval(() => {
    fetch('/get_letters', { method: 'POST' })
        .then(response => {
            if (!response.ok) throw new Error(`Ошибка сервера: ${response.status}`);
            return response.json(); // Преобразуем JSON
        })
        .then(data => {
            valueRez.innerText = data.letter; // Отображаем полученную букву
        })
        .catch(error => console.error('Ошибка:', error));
}, 5000);

// Обработчик нажатия на кнопку "Сохранить букву"
document.getElementById('save-letter').addEventListener('click',()=>{
    resultText.innerHTML += valueRez.innerHTML
})
// Обработчик нажатия на кнопку "Удалить букву"
document.getElementById('remove-letter').addEventListener('click',()=>{
    resultText.innerHTML = resultText.innerHTML.slice(0, -1)
})
// Обработчик нажатия на кнопку "Пробел"
document.getElementById('space').addEventListener('click',()=>{
    resultText.innerHTML += ' '
})
// Обработчик нажатия на кнопку "Очистить экран"
document.getElementById('clear-screen').addEventListener('click',()=>{
    clearCanvas()
})
// Обработчик нажатия на кнопку "Начать рисовать"
document.getElementById('start-drawing').addEventListener('click',()=>{
    statusDrow(true)
})
// Обработчик нажатия на кнопку "Остановить рисование"
document.getElementById('stop-drawing').addEventListener('click',()=>{
    statusDrow(false)
})
