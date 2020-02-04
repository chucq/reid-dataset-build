# import mmcv
import os, sys
import numpy as np
import random

config_file = '/data/ccq/code/detection_inference/configs/person_detection_scripts.py'
checkpoint_file = '/data/ccq/code/detection_inference/checkpoints/epoch_12.pth'
model = None
inference_detector = None
threshold = 0.4
intersection_threshold = 0.45
keep_even_no_face = False

def save_result(src,result,aimroot,cam_name,frame_index,info_out): 
    srcimg = src
    imgindex,trackinfo = result
    for person_id,info in trackinfo.items():
        aimdir = os.path.join(aimroot,str(person_id))
        if not os.path.isdir(aimdir):
            os.makedirs(aimdir)
        
        left,top,width,height = info[:4]
        snapname = '%s_P%s_V%s_F%s.png'%(cam_name,person_id,imgindex,frame_index)
        info_out.write('%s,%s,%s,%s,%s\n'%(snapname,int(left),int(top),int(left+width),int(top+height)))

        aimpath = os.path.join(aimdir,snapname)
        cimg = srcimg[max(int(info[1]),0):int(info[1]+info[3])+1,max(int(info[0]),0):int(info[0]+info[2])+1]
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


if __name__ == "__main__":
    import mmcv
    import cv2
    from PIL import Image
    import time
    
    sys.path.insert(0,'/data/ccq/code/SORT/')
    import MOTTracker as tracker

    sys.path.insert(0,'/data/ccq/code/MGN-pytorch/sdk/')
    from config import args
    import reid_feature


    init_sess()
    reid_model = reid_feature.initialize()

    #videoroot = '/data/ccq/code/detection_inference/videoData/samplevideo'
    videoroot = args.video_root
    videolist = os.listdir(videoroot)

    #aimroot = '/data/ccq/code/detection_inference/resultData/sample_track_search_out'
    aimroot = args.result_root
    if not os.path.isdir(aimroot):
        os.makedirs(aimroot)
    
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
        print('[%s] start tracklet building of %s, video fps: %s, video frames: %s, video backend: %s'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),videoname, fps, cap.get(cv2.CAP_PROP_FRAME_COUNT),cap.get(cv2.CAP_PROP_BACKEND)))

        aiminfopath = os.path.join(aimdir,'trackinfo.txt')
        info_out = open(aiminfopath,'w')

        framecount = -1
        track_ind = 0
        TK = tracker.MOTTracker(max_cosine_distance=0.3, nn_budget=200, nms_max_overlap=0.5)

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
            
            save_result(frame,result[0],aimdir,cam_name,framecount+1,info_out)

            print('[%s] track index %s at frame %s completed!'%(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),track_ind,framecount))

        info_out.close()
