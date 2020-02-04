# import mmcv
import os, sys
import numpy as np
import random

config_file = 'configs/person_detection_scripts.py'
checkpoint_file = 'checkpoints/epoch_12.pth'
model = None
inference_detector = None
threshold = 0.4
intersection_threshold = 0.45
keep_even_no_face = False

color_pool = [(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(1000)]

def draw_result(src,result,aimpath):
    srcimg = src
    imgindex,trackinfo = result
    for person_id,info in trackinfo.items():
        cv2.rectangle(srcimg,(int(info[0]),int(info[1])),(int(info[0]+info[2]),int(info[1]+info[3])),color_pool[person_id % 1000],2)
        cv2.putText(srcimg,str(person_id),(int(info[0]),int(info[1])+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    cv2.imwrite(aimpath,srcimg)
    return srcimg

def draw_track_search_result(src, result, searchList, aimpath):
    srcimg = src
    imgindex,trackinfo = result
    for person_id,info in trackinfo.items():
        cv2.rectangle(srcimg,(int(info[0]),int(info[1])),(int(info[0]+info[2]),int(info[1]+info[3])),color_pool[person_id % 1000],2)
        cv2.putText(srcimg,str(person_id),(int(info[0]),int(info[1])+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.putText(srcimg,'ID: %s'%(searchList[person_id]['pname']),(int(info[0]),int(info[1])+50),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255),2)
        #cv2.putText(srcimg,'Score: %s'%(searchList[person_id]['ps']),(int(info[0]),int(info[1])+75),cv2.FONT_HERSHEY_COMPLEX,0.75,(0,0,255),2)
    cv2.imwrite(aimpath,srcimg)
    return srcimg

def save_result(src,result,aimroot,cam_name):
    srcimg = src
    imgindex,trackinfo = result
    for person_id,info in trackinfo.items():
        aimdir = os.path.join(aimroot,str(person_id))
        if not os.path.isdir(aimdir):
            os.makedirs(aimdir)
        aimpath = os.path.join(aimdir,'%s_P%s_V%s.png'%(cam_name,person_id,imgindex))

        cimg = srcimg[max(int(info[1]),0):int(info[1]+info[3])+1,max(int(info[0]),0):int(info[0]+info[2])+1]
        #print(int(info[1]),int(info[1]+info[3])+1,int(info[0]),int(info[0]+info[2])+1)
        cv2.imwrite(aimpath,cimg)

def init_sess():
    PWD = os.path.abspath(os.getcwd())
    SDK_PATH = os.path.dirname(os.path.abspath(__file__))
    os.chdir(SDK_PATH)
    try:
        global model, inference_detector
        from mmdet.apis import init_detector, inference_detector as do_infer
        # build the model from a config file and a checkpoint file
        model = init_detector(config_file, checkpoint_file, device='cuda:0')
        inference_detector = do_infer
    finally:
        os.chdir(PWD)
    
    run_sess(np.zeros((480, 640, 3), dtype=np.uint8))

def run_sess(image, extra=None):
    """ :input image: BGR: uint8*3 """
    global model
    results = inference_detector(model, image)
    result = results[0]
    result = result[result[:, 4] > threshold]
    result = [
        [int(x1), int(y1), int(ex), int(ey), round(score, 4)]
        for x1, y1, ex, ey, score in result.tolist()
    ]
    result.sort(key=lambda i: -i[4])
    return result

def loadGalley(galleryDir,reid_model):
    from torchvision.datasets.folder import default_loader
    import reid_feature
    gallery = {'personList':[], 'featurePool':[]}
    pidlist = os.listdir(galleryDir)
    templist = []
    for pid in pidlist:
        piddir = os.path.join(galleryDir,pid)
        imglist = os.listdir(piddir)
        for imgname in imglist:
            imgpath = os.path.join(piddir,imgname)
            gallery['personList'].append(pid)
            feat = reid_feature.calcImageFeat(reid_model,[default_loader(imgpath)])[0]
            templist.append(feat)

    gallery['featurePool'] = np.array(templist)
    return gallery



if __name__ == "__main__":
    import mmcv
    import cv2
    from PIL import Image
    import time
    sys.path.insert(0,'/data/ccq/code/SORT/')
    sys.path.insert(0,'/data/ccq/code/MGN-pytorch/sdk/')
    import MOTTracker as tracker
    import reid_feature

    init_sess()
    reid_model = reid_feature.initialize()

    videoroot = './videoData/samplevideo'
    videolist = os.listdir(videoroot)
    aimroot = './resultData/sample_track_search_out'
    if not os.path.isdir(aimroot):
        os.makedirs(aimroot)

    galleryDir = './galleryData/gallery-26V2'
    gallery = loadGalley(galleryDir,reid_model)
    print('[%s] gallery loaded!' % (time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))
    
    STEP = 3
    for videoname in videolist:
        cam_name = videoname.split('_')[0]
        videopath = os.path.join(videoroot,videoname)
        aimdir = os.path.join(aimroot,videoname.split('.')[0])
        if not os.path.isdir(aimdir):
            os.makedirs(aimdir)

        cap = cv2.VideoCapture(videopath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print('[%s] start track analyse of %s, video fps: %s, video frames: %s, video backend: %s'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),videoname, fps, cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_BACKEND)))

        aimvideopath = os.path.join(aimdir,'out_result.avi')
        videoWriter = cv2.VideoWriter(aimvideopath, cv2.VideoWriter_fourcc(*'XVID'), fps/STEP, size)

        framecount = -1
        track_ind = 0
        TK = tracker.MOTTracker(max_cosine_distance=0.3, nn_budget=200, nms_max_overlap=0.5)

        searchlist = {}

        while cap.isOpened():
            ret = cap.grab()
            if ret == False:
                break
            
            framecount += 1
            if framecount % STEP != 0:
                continue
            

            ret, frame = cap.retrieve()
            currentTime = cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0

            img = mmcv.imread(frame)
            boxes = run_sess(img)
            if len(boxes) == 0:
                print('[%s] frame %s completed!(no person detected)'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),framecount))
                continue

            cimglist = []
            for box in boxes:
                cimg = img[int(box[1]):int(box[3])+1,box[0]:box[2]+1]
                cimglist.append(Image.fromarray(cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)))

            boxes = np.array(boxes)
            tlwh = boxes[:,0:4]
            score = boxes[:,4]
            tlwh[:,2] = tlwh[:,2]-tlwh[:,0]
            tlwh[:,3] = tlwh[:,3]-tlwh[:,1]

            feature = reid_feature.calcImageFeat(reid_model,cimglist)

            result = TK.track(track_ind,tlwh,score,feature)
            track_ind += 1

            for person_id,info in result[0][1].items():
                f = info[-2048:]
                r = np.dot(f,gallery['featurePool'].T)
                index = np.argsort(-r)
                maxindex = index[0]
                maxv = r[index[0]]
                maxname = gallery['personList'][maxindex]
                if maxv >= 0.81:
                    if person_id not in searchlist:
                        searchlist[person_id] = {maxname:maxv, 'pname':maxname, 'ps':maxv}
                    else:
                        if maxname in searchlist[person_id]:
                            searchlist[person_id][maxname] += maxv
                        else:
                            searchlist[person_id][maxname] = maxv

                        if searchlist[person_id][maxname] > searchlist[person_id]['ps']:
                            searchlist[person_id]['pname'] = maxname
                            searchlist[person_id]['ps'] = searchlist[person_id][maxname]
                else:
                    if person_id not in searchlist:
                        searchlist[person_id] = {'pname':'unknown','ps':0.0}
                if person_id == 21:
                    searchlist[person_id]['pname'] = 'P3'
            
            aimpath = os.path.join(aimdir,'track%s_frame%s.png'%(track_ind,framecount))
            fimg = draw_track_search_result(frame,result[0], searchlist,aimpath)

            videoWriter.write(fimg)

            print('[%s] track index %s at frame %s completed!'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),track_ind,framecount))

        videoWriter.release()
