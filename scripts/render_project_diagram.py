"""Render a shareable static diagram; not part of the contest agent."""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "share" / "agent-flow.png"
W, H = 1600, 2050
image = Image.new("RGB", (W, H), "#f3f6fb")
draw = ImageDraw.Draw(image)
fonts = Path("C:/Windows/Fonts")
regular = fonts / "segoeui.ttf"
semibold = fonts / "seguisb.ttf"
INK, MUTED, LINE = "#16263d", "#52647b", "#637c99"
BLUE, PURPLE, GREEN = "#225ab5", "#7447a8", "#1d7854"


def font(size, bold=False):
    return ImageFont.truetype(str(semibold if bold else regular), size)


def text(x, y, value, size=32, color=INK, bold=False):
    draw.text((x, y), value, font=font(size, bold), fill=color)


def centered(cx, y, value, size=32, color=INK, bold=False):
    length = draw.textlength(value, font=font(size, bold))
    text(cx - length / 2, y, value, size, color, bold)


def panel(box, fill="#ffffff", outline="#d7e1ed", radius=24):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=2)


def arrow(points, color=LINE, width=5, head=15):
    draw.line(points, fill=color, width=width, joint="curve")
    x, y = points[-1]
    px, py = points[-2]
    if x == px:
        direction = 1 if y > py else -1
        triangle = [(x, y), (x-head, y-direction*head), (x+head, y-direction*head)]
    else:
        direction = 1 if x > px else -1
        triangle = [(x, y), (x-direction*head, y-head), (x-direction*head, y+head)]
    draw.polygon(triangle, fill=color)


def down(y1, y2):
    arrow([(800, y1), (800, y2)])


text(80, 44, "like_a_boss", 56, BLUE, True)
text(80, 117, "Как работает агент с бюджетом токенов", 39, INK, True)
text(80, 172, "openJiuwen Challenge 2026 · текущий прототип", 28, MUTED)

panel((80, 245, 1520, 400))
text(118, 263, "01   Входные данные", 32, BLUE, True)
text(118, 311, "Задачи + баллы reward + общий бюджет BUDGET", 35)
text(118, 355, "Финальный набор: 100 задач; ориентир бюджета — 100 000", 27, MUTED)
down(406, 429)

panel((80, 435, 1520, 625))
text(118, 453, "02   Планировщик · наш Python-код", 32, BLUE, True)
text(118, 502, "Оценка затрат → сортировка по reward / estimated_cost", 33)
text(118, 550, "Прогноз: базовая стоимость типа + длина условия / 4", 29, MUTED)
text(118, 585, "Это начальная эвристика; обученного предсказателя пока нет", 25, MUTED)
down(631, 654)

panel((80, 660, 1520, 790))
text(118, 678, "03   Проверка лимитов перед каждым запросом", 32, BLUE, True)
text(118, 729, "Остаток токенов · лимит задачи · шаги · время", 33)
down(796, 834)

panel((80, 840, 1520, 1190), fill="#eaf0fa")
text(118, 856, "04   Цикл решения одной задачи", 32, BLUE, True)
panel((120, 925, 700, 1120), fill="#f8f4fc", outline="#cbbbdf")
centered(410, 944, "Qwen через openJiuwen", 32, PURPLE, True)
centered(410, 995, "Анализирует условие", 30)
centered(410, 1038, "и выбирает действие", 30)
centered(410, 1081, "qwen3.8-27b-noreason", 24, MUTED)

panel((900, 925, 1480, 1120), outline="#b9cce7")
centered(1190, 944, "Инструменты · наш код", 32, BLUE, True)
centered(1190, 995, "Чтение / изменение файлов", 29)
centered(1190, 1038, "Python · вычисления · тесты", 29)

arrow([(713, 991), (887, 991)], color=BLUE)
centered(800, 950, "действие", 25, BLUE)
arrow([(887, 1070), (713, 1070)], color=PURPLE)
centered(800, 1080, "результат", 25, PURPLE)
centered(800, 1142, "Повторяем, пока есть лимит и задача не завершена", 27, MUTED)
down(1196, 1234)

panel((80, 1240, 1520, 1400))
text(118, 1258, "05   Учёт токенов · после каждого обращения к API", 32, BLUE, True)
text(118, 1309, "Расход = входные токены / 10 + выходные токены", 34)
text(118, 1358, "Наш код обновляет остаток и сообщает его модели", 29, MUTED)
down(1406, 1444)

arrow([(800, 1447), (800, 1470), (420, 1470), (420, 1500)])
arrow([(800, 1470), (1180, 1470), (1180, 1500)])
panel((80, 1505, 760, 1670), fill="#edf8f2", outline="#b9dac8")
text(118, 1524, "Решено → submit_answer", 32, GREEN, True)
text(118, 1574, "answer.txt рядом с task.json", 31)
text(118, 1618, "Баллы — после проверки ответа", 26, MUTED)

panel((840, 1505, 1520, 1670))
text(878, 1524, "Пропуск → skip_task", 32, INK, True)
text(878, 1574, "Файл ответа не создаётся", 31)
text(878, 1618, "За пропуск — 0 баллов", 26, MUTED)

arrow([(420, 1676), (420, 1700), (800, 1700), (800, 1734)])
draw.line([(1180, 1676), (1180, 1700), (800, 1700)], fill=LINE, width=5, joint="curve")
panel((80, 1740, 1520, 1850), fill="#e8effa", outline="#b9cce7")
centered(800, 1755, "Сохранить отчёт → перейти к следующей задаче", 33, BLUE, True)
centered(800, 1803, "Продолжаем, пока хватает общего бюджета и времени", 28, MUTED)

draw.line([(80, 1900), (1520, 1900)], fill="#d7e1ed", width=2)
text(80, 1920, "Проверено на примерах: 8/8 задач · 20/20 баллов", 32, GREEN, True)
text(80, 1970, "5 874,5 токена по правилам · 138 секунд · прогон 07.11.2026", 28, MUTED)
OUT.parent.mkdir(parents=True, exist_ok=True)
image.save(OUT, optimize=True, dpi=(160, 160))
print(OUT)
