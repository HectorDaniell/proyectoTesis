import cv2
import mediapipe as mp
from extract_landmarks import extract_landmarks_from_frame  # Importamos la función de otro archivo

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture("/Users/danieloviedo/Library/CloudStorage/OneDrive-Personal/Documentos/Tesis/proyectoTesis/data/raw/jump/jump_008.mp4")
#cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_holistic.Holistic(
     static_image_mode=False,
     model_complexity=1) as holistic:

     while True:
          ret, frame = cap.read()
          if ret == False:
               break

          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = holistic.process(frame_rgb)

          # Postura
          mp_drawing.draw_landmarks(
               frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
               mp_drawing.DrawingSpec(color=(128, 0, 255), thickness=2, circle_radius=1),
               mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

          # Llamamos a la función para extraer y guardar los puntos clave
          extract_landmarks_from_frame(results)

          frame = cv2.flip(frame, 1)
          cv2.imshow("Frame", frame)
          if cv2.waitKey(1) & 0xFF == 27:
               break

cap.release()
cv2.destroyAllWindows()