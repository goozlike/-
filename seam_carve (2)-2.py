#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
from skimage import io

def count_gradients(img_bright):
    gradient_x = np.zeros(img_bright.shape)
    gradient_y = np.zeros(img_bright.shape)
    for i in range(1, len(img_bright[0]) - 1):
        gradient_x[:, i] = img_bright[:, i + 1] - img_bright[:, i - 1]
    for i in range(1, len(img_bright) - 1):
        gradient_y[i, :] = img_bright[i + 1, :] - img_bright[i - 1, :]
    gradient_x[:, 0] = img_bright[:, 1] - img_bright[:, 0]
    gradient_x[:, -1] = img_bright[:, -1] - img_bright[:, -2]
    gradient_y[0, :] = img_bright[1, :] - img_bright[0, :]
    gradient_y[-1, :] = img_bright[-1, :] - img_bright[-2, :]        
    return gradient_x, gradient_y

def find_vert_seam(img_energy):
    new_mat = np.copy(img_energy)
    for y in range(1, len(new_mat)):
        for x in range(len(new_mat[0])):
            if x == 0:
                new_mat[y][x] += min(new_mat[y-1][0], new_mat[y-1][1])
            elif x == len(new_mat[0]) - 1:
                new_mat[y][x] += min(new_mat[y-1][len(new_mat[0]) - 1], new_mat[y-1][len(new_mat[0]) - 2])
            else:
                new_mat[y][x] += min(new_mat[y-1][x-1],new_mat[y-1][x],new_mat[y-1][x+1])
    seam = []
    x_seam = np.argmin(new_mat[-1])
    seam.append(x)
    for i in range(len(new_mat) - 1, -1, -1):
        if x_seam == 0:
            new_x = np.argmin(new_mat[i, 0 : 2])
        elif x_seam == len(new_mat[0]) - 1:
            new_x = np.argmin(new_mat[i, len(new_mat[0]) - 2 :]) + len(new_mat[0]) - 2
        else:
            new_x = np.argmin(new_mat[i, x_seam - 1 : x_seam + 2]) + x_seam - 1
        seam.append(new_x)
        x_seam = new_x
    seam.reverse()
    seam.pop()
    return seam

def horizontal_shrink(img, img_energy, mask):
    seam = find_vert_seam(img_energy)
    new_img = []
    new_mask = []
    seam_mask = np.zeros(img_energy.shape)    
    for i in range(len(seam)):
        img_string = np.delete(img[i], seam[i],0)
        mask_string = np.delete(mask[i], seam[i],0)
        new_img.append(img_string)
        new_mask.append(mask_string)
        seam_mask[i, seam[i]] = 1
    return np.array(new_img), np.array(new_mask), seam_mask

def horizontal_expand(img, img_energy, mask):
    seam = find_vert_seam(img_energy)
    new_image = []
    new_mask = []
    seam_mask = np.zeros(img_energy.shape)
    for i in range(len(seam)):
        if seam[i] != len(img_energy[0]) - 1:
            img_string = np.insert(img[i], seam[i] + 1, (img[i][seam[i]] + img[i][seam[i] + 1]) / 2, 0)
            new_image.append(img_string)
        else:
            img_string = np.insert(img[i], 
                                   seam[i] + 1, img[i][seam[i]], 0)
            new_image.append(img_string)
        seam_mask[i, seam[i]] = 1
        mask_string = np.insert(mask[i], seam[i] + 1, 0, 0)
        new_mask.append(mask_string)
        new_mask[i][seam[i]] += 256 * len(img_energy) * len(img_energy[0])
    return np.array(new_image), np.array(new_mask), np.array(seam_mask)

def seam_carve(img, func, mask = None):
    img_bright = img[:,:,0]*0.299 + img[:,:,1]*0.587 + img[:,:,2]*0.114
    gradient_x, gradient_y = count_gradients(img_bright)
    img_energy = np.sqrt(np.square(gradient_x) + np.square(gradient_y))
    if np.any(mask) != None:
        img_energy = len(img_energy) * len(img_energy[0]) * 256 * mask + img_energy
    else:
        mask = np.zeros(img_energy.shape)


    if func == "horizontal shrink":
        return horizontal_shrink(img, img_energy, mask
                                )
    elif func == "vertical shrink":
        img_T = np.transpose(img, axes=(1,0,2))
        img_energy_T = np.transpose(img_energy)
        mask_T = np.transpose(mask)
        new_img_T, new_mask_T, seam_mask_T = horizontal_shrink(img_T, img_energy_T, mask_T)
        return np.transpose(new_img_T, axes=(1,0,2)), np.transpose(new_mask_T), np.transpose(seam_mask_T)
    
    elif func == "horizontal expand":
        return horizontal_expand(img, img_energy, mask)
    
    elif func == "vertical expand":
        img_T = np.transpose(img, axes=(1,0,2))
        img_energy_T = np.transpose(img_energy)
        mask_T = np.transpose(mask)
        new_img_T, new_mask_T, seam_mask_T = horizontal_expand(img_T, img_energy_T, mask_T)
        return np.transpose(new_img_T, axes=(1,0,2)), np.transpose(new_mask_T), np.transpose(seam_mask_T)
    

#img = io.imread(r'/Users/goozlike/Downloads/public_tests-2/05_test_img_input/img.png', plugin='matplotlib')
#mask = io.imread(r'/Users/goozlike/Downloads/public_tests-2/05_test_img_input/mask.png', plugin='matplotlib')
#ans = np.copy(img)
#n_mask = np.zeros((mask.shape[0], mask.shape[1]))
#for i in range(mask.shape[0]):
#    for j in range(mask.shape[1]):
#        if mask[i,j,0] > 0:
#            n_mask[i][j] = -1
#        elif mask[i,j,1] > 0:
#            n_mask[i][j] = 1
#for i in range(100):
#    print(i)
    
#    ans, n_mask, seam = seam_carve(ans, "horizontal expand", n_mask)



#io.imshow(ans)



# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




