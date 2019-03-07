import cv2

img = cv2.imread('../img/blank_500.jpg')

# sans-serif small
cv2.putText(img, "Plain", (50, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0,0))            
# sans-serif normal
cv2.putText(img, "Simplex", (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0,0))        
# sans-serif bold
cv2.putText(img, "Duplex", (50, 110), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0,0))         
# sans-serif normall X2  ---①
cv2.putText(img, "Simplex", (200, 110), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,250)) 

# serif small
cv2.putText(img, "Complex Small", (50, 180), cv2.FONT_HERSHEY_COMPLEX_SMALL, \
            1, (0, 0,0))   
# serif normal
cv2.putText(img, "Complex", (50, 220), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0,0))
# serif bold
cv2.putText(img, "Triplex", (50, 260), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0,0))               
# serif normal X2  ---②
cv2.putText(img, "Complex", (200, 260), cv2.FONT_HERSHEY_TRIPLEX, 2, (0,0,255))               

# hand-wringing sans-serif
cv2.putText(img, "Script Simplex", (50, 330), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, \
            1, (0, 0,0)) 
# hand-wringing serif
cv2.putText(img, "Script Complex", (50, 370), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, \
            1, (0, 0,0)) 

# sans-serif + italic ---③
cv2.putText(img, "Plain Italic", (50, 430), \
            cv2.FONT_HERSHEY_PLAIN | cv2.FONT_ITALIC, 1, (0, 0,0)) 
# sarif + italic
cv2.putText(img, "Complex Italic", (50, 470), \
            cv2.FONT_HERSHEY_COMPLEX | cv2.FONT_ITALIC, 1, (0, 0,0)) 

cv2.imshow('draw text', img)
cv2.waitKey()
cv2.destroyAllWindows()