import torch

# SR : Segmentation Result
# GT : Ground Truth

def get_accuracy(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT > threshold).float()
    corr = SR==GT
    # tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    # acc = float(corr)/float(tensor_size)
    acc = corr.float().mean().item()
    return acc

def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = (SR > threshold).float()
    GT = (GT > threshold).float()

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).float()+(GT==1).float())==2
    FN = ((SR==0).float()+(GT==1).float())==2

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)     
    
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT > threshold).float()

    # TN : True Negative
    # FP : False Positive
    TN = (((SR==0).float()+(GT==0).float())==2).float()
    FP = (((SR==1).float()+(GT==0).float())==2).float()

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP

def get_precision(SR,GT,threshold=0.5):
    SR = (SR > threshold).float()
    GT = (GT > threshold).float()
    # import ipdb; ipdb.set_trace()
    # TP : True Positive
    # FP : False Positive
    TP = (((SR==1).float()+(GT==1).float())==2).float()
    FP = (((SR==1).float()+(GT==0).float())==2).float()

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC

def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1

def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = (SR > threshold).float()
    GT = (GT > threshold).float()
    Inter = torch.sum((SR+GT)==2)
    Union = torch.sum((SR+GT)>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS

def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR =( SR > threshold).float()
    GT = (GT > threshold).float()

    Inter = torch.sum((SR+GT)==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC



