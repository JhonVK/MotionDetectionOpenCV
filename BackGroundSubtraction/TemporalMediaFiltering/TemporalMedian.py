import numpy as np
import cv2

video_source='C:/Users/joaov/OneDrive/MotionDetectionOpenCV/Assets/videos/Cars.mp4'   #FONTE DO VIDEO
video_out='C:/Users/joaov/OneDrive/MotionDetectionOpenCV/Assets/videos/results/filtragem_mediana_temporal.avi' #SAIDA DO VIDEO

cap=cv2.VideoCapture(video_source) #LEITURA DO VIDEO(CAPTURA)

hasFrame, frame= cap.read() #HASFRAME VAI INDICAR SE TEM FRAME, FIRSTFRAME É O PRIMEIRO FRAME
print(hasFrame, frame.shape)

fourcc= cv2.VideoWriter_fourcc(*'XVID') #GRAVAR O VIDEO DE RESULTADO fourcc é o codec (xvid é o formato para videos AVI)

writer=cv2.VideoWriter(video_out, fourcc, 25, (frame.shape[1], frame.shape[0]), True ) #writer para gravar o video video_out vai ser onde vamos salvar, fourcc é o codec, 25 é o numero de frames p segundo, frame.shape é a dimensão do video. True significa que o video é colorido

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))#VAI MOSTRAR QUANTOS FRAMES A IMAGEM TEM
#PRECISAMOS EXTRAIR 25 FRAMES DESSES 3000 (ALEATORIAMENTE)

framesIds= cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25) #GERANDO OS FRAMES

#cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
#hasFrame, frame= cap.read() #LEITURA DO FRAME ESPECÍFICO
#cv2.imshow('test', frame)
#cv2.waitKey(0)

frames=[]
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame= cap.read()
    frames.append(frame)

print(np.asarray(frames).shape)

for frame in frames:
    cv2.imshow('frame', frame)
    cv2.waitKey(0)