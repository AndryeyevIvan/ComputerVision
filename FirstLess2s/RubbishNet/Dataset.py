import cv2
import os
import time

klasy = {
    ord('1'): "plastic",
    ord('2'): "metal",
    ord('3'): "other"
}

korin = "dataset"
interval = 0.3

for nazva in klasy.values():
    os.makedirs(os.path.join(korin, nazva), exist_ok=True)

kamera = cv2.VideoCapture(0)

potocnyi_klas = None
ostannyi_chas = 0

print("Натисни 1 - plastic | 2 - metal | 3 - other")
print("ESC - вихід")

while True:
    ret, kadr = kamera.read()
    if not ret:
        break

    pokaz = kadr.copy()

    if potocnyi_klas:
        cv2.putText(pokaz, f"Запис: {potocnyi_klas}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        if time.time() - ostannyi_chas > interval:
            imya = f"{int(time.time()*1000)}.jpg"
            shliakh = os.path.join(korin, potocnyi_klas, imya)
            cv2.imwrite(shliakh, kadr)
            ostannyi_chas = time.time()

    cv2.imshow("Zbir_danyh", pokaz)

    klavisha = cv2.waitKey(1)

    if klavisha in klasy:
        potocnyi_klas = klasy[klavisha]
        print("Поточний клас:", potocnyi_klas)

    if klavisha == 27:
        break

kamera.release()
cv2.destroyAllWindows()
