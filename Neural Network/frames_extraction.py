from ultralytics import YOLO
import cv2
import shutil
import math
import os

model = YOLO('../Armored_model.pt')

classNames = ['pistol', 'bat', 'sword', 'epee', 'axe', 'knife']

video_path = r"Ваш путь к видео"

cap = cv2.VideoCapture(video_path)
cv2.waitKey(0)

save_path = r"Путь для сохранения. Можно использовать уже созданную папку 'predictions'"

if os.path.isfile(save_path):
    shutil.rmtree(save_path)
    os.mkdir(save_path)

unique_id = set()


while cap.isOpened():

    success, frame = cap.read()

    if success:
        # Запуск YOLO для каждого кадра
        results = model.track(frame, persist=True)
        c = 0
        if results[0].boxes.id is not None:
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            boxes_with_conf = results[0].boxes
            for box, conf_box, id in zip(boxes, boxes_with_conf, ids):
                # Проверка id на уникальность
                int_id = int(id)

                conf = math.ceil((conf_box.conf[0] * 100)) / 100
                cls = int(conf_box.cls[0])

                if int_id not in unique_id:
                    unique_id.add(int_id)

                    # Сохранение кадра под уникальным id
                    filename = f"{int_id}.jpg"
                    filepath = os.path.join(save_path, filename)
                    cv2.imwrite(filepath, frame)

                # Нанесение границ, класса и увренности(confidence score)
                cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (85, 45, 255), 2, lineType=cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"{classNames[cls]} {conf}",
                    (box[0], box[1]),
                    0,
                    0.9,
                    [85, 45, 255],
                    2,
                    lineType=cv2.LINE_AA
                )
                cv2.imshow("Detected Frame", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        cv2.imshow("Detected Frame", frame)
        # Нажмие q, чтобы прервать работу
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
