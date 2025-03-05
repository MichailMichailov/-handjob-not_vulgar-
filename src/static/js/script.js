// Отримуємо елементи DOM
const valueRez = document.getElementById('tmpRez')  // Елемент, який відображає результат розпізнавання
const resultText = document.getElementById('full-text')  // Елемент, куди додається текст

// Функція для очищення полотна
function clearCanvas() {
    // Відправляємо запит на очищення полотна на сервер
    fetch('/clear_canvas', { method: 'POST' })
        .then(response => console.log('Полотно очищено')) // Успішне виконання
        .catch(error => console.error('Помилка:', error)); // Помилка при запиті
}

function statusDrow(value){
    fetch('/set_is_write', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ is_write: value }) // Можна відправити будь-яке значення, якщо потрібно
    })
    .then(response => {
        if (!response.ok){
            console.error("Помилка при відправці запиту");
        }
    })
    .catch(error => console.error("Помилка мережі:", error));
}

// Слухаємо натискання клавіш
window.addEventListener('keydown', function(event) {
    if (event.key === 'c' || event.key === 'C') {  // Якщо натиснута клавіша C
        clearCanvas(); // Очищаємо полотно
    }else if (event.key === 's' || event.key === 'S') {  // Якщо натиснута клавіша S
        resultText.innerHTML += valueRez.innerHTML
    }else if (event.key === ' ' || event.key === ' ') {  // Якщо натиснута клавіша пробіл
        resultText.innerHTML += ' '
    }else if (event.key === 'd' || event.key === 'D') {  // Якщо натиснута клавіша D
        resultText.innerHTML = resultText.innerHTML.slice(0, -1)
    }
});

// Отримуємо дані з сервера з інтервалом 5 секунд (можна скоротити або збільшити)
setInterval(() => {
    fetch('/get_letters', { method: 'POST' })
        .then(response => {
            if (!response.ok) throw new Error(`Помилка сервера: ${response.status}`);
            return response.json(); // Перетворюємо JSON
        })
        .then(data => {
            valueRez.innerText = data.letter; // Відображаємо отриману букву
        })
        .catch(error => console.error('Помилка:', error));
}, 5000);

// Обробник натискання на кнопку "Зберегти букву"
document.getElementById('save-letter').addEventListener('click',()=>{
    resultText.innerHTML += valueRez.innerHTML
})

// Обробник натискання на кнопку "Видалити букву"
document.getElementById('remove-letter').addEventListener('click',()=>{
    resultText.innerHTML = resultText.innerHTML.slice(0, -1)
})

// Обробник натискання на кнопку "Пробіл"
document.getElementById('space').addEventListener('click',()=>{
    resultText.innerHTML += ' '
})

// Обробник натискання на кнопку "Очистити екран"
document.getElementById('clear-screen').addEventListener('click',()=>{
    clearCanvas()
})

// Обробник натискання на кнопку "Розпочати малювання"
document.getElementById('start-drawing').addEventListener('click',()=>{
    statusDrow(true)
})

// Обробник натискання на кнопку "Зупинити малювання"
document.getElementById('stop-drawing').addEventListener('click',()=>{
    statusDrow(false)
})
