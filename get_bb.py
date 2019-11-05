from common import *
pkl_path = 'datasets/DrugCharSeg/train_ocr_label.pkl'
image_dir = 'datasets/DrugCharSeg/images/'
seg_paths = glob('./dataset/*/*.png')

def get_seg(name):
    for seg_path in seg_paths:
        if os.path.basename(seg_path) == name:
            mask = cv2.imread(seg_path, 0)
    w = mask.shape[1]
    return mask[:,w//2:]
def get_image_path(name):
    img_path = os.path.join(image_dir, '1', name)
    if os.path.exists(img_path):
        return img_path
    else:
        img_path = os.path.join(image_dir, '2', name)
        assert os.path.exists(img_path), img_path
        return img_path



with open(pkl_path, 'rb') as f:
    all_data = pickle.load(f)
texts = []
for data in all_data.values():
    for bb in data['bboxes']:
        texts.append(bb['text'])
texts = list(set(texts))
text_to_int = {text:i+1 for i, text in enumerate(texts)}
# all_text = [data['text'] for text in ]
os.makedirs('dataset/ocr_segment/A', exist_ok=1)
os.makedirs('dataset/ocr_segment/B', exist_ok=1)

for path, data in all_data.items():
    name = os.path.basename(path)
    img_path = get_image_path(name)
    img = cv2.imread(img_path)
    mask = get_seg(name)
    h, w = mask.shape[:2]
    mask = cv2.resize(mask, (w, h))
    mask_zero = np.zeros([h, w, len(text_to_int)+1], 'uint8')
    
    for bb in data['bboxes']:
        pad = np.zeros([h, w])
        cv2.drawContours(pad, np.array([bb['bbox']]), -1, 255, -1)
        char_pad = mask*pad
        cls_id = text_to_int[bb['text']]
        mask_zero[:,:,cls_id] = mask_zero[:,:,cls_id] + char_pad.astype(mask_zero.dtype)
    
    
    out_path_a = os.path.join('dataset/ocr_segment/A', name)
    out_path_b = os.path.join('dataset/ocr_segment/B', name)
    cv2.imwrite(out_path_a, img)
    cv2.imwrite(out_path_b, np.argmax(mask_zero, -1).astype('uint8'))
    # plt.figure()
    # os.makedirs('cache', exist_ok=1)
    # plt.imshow(np.argmax(mask_zero, -1))
    # plt.savefig(os.path.join('cache',name))
    # plt.close()
    # break 

print(len(texts))