import cv2
import numpy as np

def laplacian_artifacts(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    _, mask = cv2.threshold(np.abs(laplacian), 20, 255, cv2.THRESH_BINARY)
    artifact_percentage = (np.sum(mask > 0) / mask.size) * 100
    mask_colored = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return mask_colored, artifact_percentage

def gaussian_blur_artifacts(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian_blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, gaussian_blur)
    _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    artifact_percentage = (np.sum(mask > 0) / mask.size) * 100
    mask_colored = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    return mask_colored, artifact_percentage

def main():
    cap = cv2.VideoCapture(0)  # Capture from webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply Gaussian and Laplacian artifact detection methods
        laplacian_mask, laplacian_percentage = laplacian_artifacts(frame)
        gaussian_mask, gaussian_percentage = gaussian_blur_artifacts(frame)
        
        # Calculate weighted score
        weighted_score = 0.75 * gaussian_percentage + 0.25 * laplacian_percentage
        
               # Determine classification based on weighted score
      # Green for likely real
        
        # Combine masks for overlay
        combined_mask = cv2.addWeighted(laplacian_mask, 0.5, gaussian_mask, 0.5, 0)
        
        # Overlay combined mask on original frame
        overlayed_frame = cv2.addWeighted(frame, 0.7, combined_mask, 0.3, 0)
        
        # Display the percentages on the frame
        font = cv2.FONT_HERSHEY_SIMPLEX
        text1 = f"Laplacian: {laplacian_percentage:.2f}%"
        text2 = f"Gaussian: {gaussian_percentage:.2f}%"
        text3 = f"Score: {weighted_score:.2f}"
        
        cv2.putText(overlayed_frame, text1, (10, 30), font, 0.5, (0, 0, 255), 1)
        cv2.putText(overlayed_frame, text2, (10, 50), font, 0.5, (0, 0, 255), 1)
        cv2.putText(overlayed_frame, text3, (10, 70), font, 0.5, (0, 0, 255), 1)
       
        # Display metadata
        metadata_text = f"Resolution: {frame.shape[1]}x{frame.shape[0]}"
        cv2.putText(overlayed_frame, metadata_text, (10, frame.shape[0] - 10), font, 0.5, (255, 0, 0), 1)

        # Display the frame with overlay and text
        cv2.imshow('Real Camera Footage', overlayed_frame)
        
        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
