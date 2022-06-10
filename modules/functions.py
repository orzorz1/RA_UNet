
def dice_coeff(pred, target):
    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1) #Flatten
    m2 = target.view(num, -1) #Flatten
    intersection = (m1 * m2).sum()

    return (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

def dice_loss(pred, target):
    smooth = 1.0
    num = pred.size(0)
    m1 = pred.view(num, -1).float() #Flatten
    m2 = target.view(num, -1).float() #Flatten
    intersection = (m1 * m2).sum()
    dice_coeff = (2.0 * intersection + smooth) / (m1.sum() + m2.sum() + smooth)
    return 1-dice_coeff