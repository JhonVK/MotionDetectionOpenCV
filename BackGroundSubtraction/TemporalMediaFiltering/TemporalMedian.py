import numpy as np
import cv2

video_source='C:/Users/joaov/OneDrive/MotionDetectionOpenCV/Assets/videos/Cars.mp4'   #FONTE DO VIDEO
video_out='C:/Users/joaov/OneDrive/MotionDetectionOpenCV/Assets/videos/results/filtragem_mediana_temporal.avi' #SAIDA DO VIDEO

cap=cv2.VideoCapture(video_source) #LEITURA DO VIDEO(CAPTURA)

hasFrame, frame= cap.read() #hasFrame indica se o quadro foi lido com sucesso, frame contém o primeiro quadro

fourcc= cv2.VideoWriter_fourcc(*'XVID') #GRAVAR O VIDEO DE RESULTADO fourcc é o codec (xvid é o formato para videos AVI)

writer=cv2.VideoWriter(video_out, fourcc, 250, (frame.shape[1], frame.shape[0]), True ) #writer para gravar o video video_out vai ser onde vamos salvar, fourcc é o codec, 25 é o numero de frames p segundo, frame.shape é a dimensão do video. True significa que o video é colorido

#print(cap.get(cv2.CAP_PROP_FRAME_COUNT))#VAI MOSTRAR QUANTOS FRAMES A IMAGEM TEM
#PRECISAMOS EXTRAIR 25 FRAMES DESSES 3000 (ALEATORIAMENTE)

framesIds= cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=250) #GERANDO OS FRAMES

#cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)
#hasFrame, frame= cap.read() #LEITURA DO FRAME ESPECÍFICO
#cv2.imshow('test', frame)
#cv2.waitKey(0)

frames=[]
for fid in framesIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    hasFrame, frame= cap.read()
    frames.append(frame)

#print(np.asarray(frames).shape)


#calculando a mediana (pixel central) das imagens
medianFrame=np.median(frames, axis=0).astype(dtype=np.uint8)

cv2.imshow('frame medio', medianFrame)
cv2.waitKey(0)


#Resetando o cap para iniciar no primeiro frame
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

#convertendo o median frame em tons de cinza (processamento se torna mais otimizado)
grayMedianFrame=cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
cv2.imshow('gray', grayMedianFrame)
cv2.waitKey(0)

while (True):
    hasFrame, frame = cap.read()

    if not hasFrame:
        print('Error')
        break

    frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    dframe = cv2.absdiff(frameGray, grayMedianFrame)

    #th, dframe = cv2.threshold(dframe, 70, 255, cv2.THRESH_BINARY) 
    #binarizando a saida
    th, dframe = cv2.threshold(dframe, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    print(th)

    cv2.imshow('frame', dframe)
    writer.write(dframe)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

writer.release()
cap.release()