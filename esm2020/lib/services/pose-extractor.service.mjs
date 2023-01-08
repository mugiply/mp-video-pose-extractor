import { EventEmitter, Injectable } from '@angular/core';
import { HAND_CONNECTIONS, Holistic, POSE_CONNECTIONS, POSE_LANDMARKS, POSE_LANDMARKS_LEFT, POSE_LANDMARKS_RIGHT, } from '@mediapipe/holistic';
import { drawConnectors, drawLandmarks, lerp } from '@mediapipe/drawing_utils';
import * as i0 from "@angular/core";
/**
 * MediaPipe を用いて動画からポーズを抽出するためのサービス
 *
 * ※ シングルトンなサービスではないため、Component で providers に指定して使用することを想定
 */
export class PoseExtractorService {
    constructor() {
        this.onResultsEventEmitter = new EventEmitter();
        this.init();
    }
    getPosePreviewMediaStream() {
        if (!this.posePreviewCanvasElement)
            return;
        return this.posePreviewCanvasElement.captureStream();
    }
    getHandPreviewMediaStream() {
        if (!this.handPreviewCanvasElement)
            return;
        return this.handPreviewCanvasElement.captureStream();
    }
    async onVideoFrame(videoElement) {
        if (!this.holistic)
            return;
        if (this.posePreviewCanvasElement) {
            this.posePreviewCanvasElement.width = videoElement.videoWidth;
            this.posePreviewCanvasElement.height = videoElement.videoHeight;
        }
        if (this.handPreviewCanvasElement) {
            this.handPreviewCanvasElement.width = videoElement.videoWidth;
            this.handPreviewCanvasElement.height = videoElement.videoHeight;
        }
        await this.holistic.send({ image: videoElement });
    }
    init() {
        this.posePreviewCanvasElement = document.createElement('canvas');
        this.posePreviewCanvasContext =
            this.posePreviewCanvasElement.getContext('2d') || undefined;
        this.handPreviewCanvasElement = document.createElement('canvas');
        this.handPreviewCanvasContext =
            this.handPreviewCanvasElement.getContext('2d') || undefined;
        this.holistic = new Holistic({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`;
            },
        });
        this.holistic.setOptions({
            selfieMode: false,
            modelComplexity: 1,
            smoothLandmarks: true,
            enableSegmentation: false,
            smoothSegmentation: true,
            minDetectionConfidence: 0.6,
            minTrackingConfidence: 0.6,
        });
        this.holistic.onResults((results) => {
            this.onResults(results);
        });
    }
    onResults(results) {
        if (!this.posePreviewCanvasElement ||
            !this.posePreviewCanvasContext ||
            !this.holistic)
            return;
        // 描画用に不必要なランドマークを除去
        let poseLandmarks = [];
        if (results.poseLandmarks) {
            poseLandmarks = JSON.parse(JSON.stringify(results.poseLandmarks));
            this.removeElements(poseLandmarks, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22]);
        }
        // キャンバスを塗りつぶし
        this.posePreviewCanvasContext.save();
        this.posePreviewCanvasContext.clearRect(0, 0, this.posePreviewCanvasElement.width, this.posePreviewCanvasElement.height);
        // 検出に使用したフレーム画像を描画
        this.posePreviewCanvasContext.drawImage(results.image, 0, 0, this.posePreviewCanvasElement.width, this.posePreviewCanvasElement.height);
        // 肘と手をつなぐ線を描画
        this.posePreviewCanvasContext.lineWidth = 5;
        if (poseLandmarks) {
            if (results.rightHandLandmarks) {
                this.posePreviewCanvasContext.strokeStyle = 'white';
                this.connect(this.posePreviewCanvasContext, [
                    [
                        poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW],
                        results.rightHandLandmarks[0],
                    ],
                ]);
            }
            if (results.leftHandLandmarks) {
                this.posePreviewCanvasContext.strokeStyle = 'white';
                this.connect(this.posePreviewCanvasContext, [
                    [
                        poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW],
                        results.leftHandLandmarks[0],
                    ],
                ]);
            }
        }
        // ポーズのプレビューを描画
        if (poseLandmarks) {
            drawConnectors(this.posePreviewCanvasContext, poseLandmarks, POSE_CONNECTIONS, { color: 'white' });
            drawLandmarks(this.posePreviewCanvasContext, Object.values(POSE_LANDMARKS_LEFT).map((index) => poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(255,138,0)' });
            drawLandmarks(this.posePreviewCanvasContext, Object.values(POSE_LANDMARKS_RIGHT).map((index) => poseLandmarks[index]), { visibilityMin: 0.65, color: 'white', fillColor: 'rgb(0,217,231)' });
        }
        // 手のプレビューを描画
        drawConnectors(this.posePreviewCanvasContext, results.rightHandLandmarks, HAND_CONNECTIONS, { color: 'white' });
        drawLandmarks(this.posePreviewCanvasContext, results.rightHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(0,217,231)',
            lineWidth: 2,
            radius: (data) => {
                return lerp(data.from.z, -0.15, 0.1, 10, 1);
            },
        });
        drawConnectors(this.posePreviewCanvasContext, results.leftHandLandmarks, HAND_CONNECTIONS, { color: 'white' });
        drawLandmarks(this.posePreviewCanvasContext, results.leftHandLandmarks, {
            color: 'white',
            fillColor: 'rgb(255,138,0)',
            lineWidth: 2,
            radius: (data) => {
                return lerp(data.from.z, -0.15, 0.1, 10, 1);
            },
        });
        // 手の領域のみのプレビューを生成
        if (this.handPreviewCanvasContext && this.handPreviewCanvasElement) {
            const HAND_PREVIEW_ZOOM = 3;
            const handPreviewBaseY = this.handPreviewCanvasElement.height / 2;
            this.handPreviewCanvasContext.clearRect(0, 0, this.handPreviewCanvasElement.width, this.handPreviewCanvasElement.height);
            if (results.rightHandLandmarks) {
                const rect = this.getRectByLandmarks(results.rightHandLandmarks, results.image.width, results.image.height);
                let handPreviewX = 0;
                let handPreviewY = handPreviewBaseY - (rect[3] * HAND_PREVIEW_ZOOM) / 2;
                this.handPreviewCanvasContext.drawImage(this.posePreviewCanvasElement, rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10, handPreviewX, handPreviewY, rect[2] * HAND_PREVIEW_ZOOM, rect[3] * HAND_PREVIEW_ZOOM);
            }
            if (results.leftHandLandmarks) {
                const rect = this.getRectByLandmarks(results.leftHandLandmarks, results.image.width, results.image.height);
                let handPreviewX = this.handPreviewCanvasElement.width - rect[2] * HAND_PREVIEW_ZOOM;
                let handPreviewY = handPreviewBaseY - (rect[3] * HAND_PREVIEW_ZOOM) / 2;
                this.handPreviewCanvasContext.drawImage(this.posePreviewCanvasElement, rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10, handPreviewX, handPreviewY, rect[2] * HAND_PREVIEW_ZOOM, rect[3] * HAND_PREVIEW_ZOOM);
            }
        }
        // イベントを送出
        this.onResultsEventEmitter.emit({
            mpResults: results,
            posePreviewImageDataUrl: this.posePreviewCanvasElement.toDataURL('image/jpeg'),
        });
        // 完了
        this.posePreviewCanvasContext.restore();
    }
    connect(ctx, connectors) {
        const canvas = ctx.canvas;
        for (const connector of connectors) {
            const from = connector[0];
            const to = connector[1];
            if (from && to) {
                if (from.visibility &&
                    to.visibility &&
                    (from.visibility < 0.1 || to.visibility < 0.1)) {
                    continue;
                }
                ctx.beginPath();
                ctx.moveTo(from.x * canvas.width, from.y * canvas.height);
                ctx.lineTo(to.x * canvas.width, to.y * canvas.height);
                ctx.stroke();
            }
        }
    }
    removeElements(landmarks, elements) {
        for (const element of elements) {
            delete landmarks[element];
        }
    }
    getRectByLandmarks(landmarks, width, height) {
        const leftHandLandmarksX = landmarks.map((landmark) => landmark.x * width);
        const leftHandLandmarksY = landmarks.map((landmark) => landmark.y * height);
        const minX = Math.min(...leftHandLandmarksX);
        const maxX = Math.max(...leftHandLandmarksX);
        const minY = Math.min(...leftHandLandmarksY);
        const maxY = Math.max(...leftHandLandmarksY);
        return [minX, minY, maxX - minX, maxY - minY];
    }
}
PoseExtractorService.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseExtractorService, deps: [], target: i0.ɵɵFactoryTarget.Injectable });
PoseExtractorService.ɵprov = i0.ɵɵngDeclareInjectable({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseExtractorService });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseExtractorService, decorators: [{
            type: Injectable
        }], ctorParameters: function () { return []; } });
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1leHRyYWN0b3Iuc2VydmljZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3Byb2plY3RzL25neC1tcC1wb3NlLWV4dHJhY3Rvci9zcmMvbGliL3NlcnZpY2VzL3Bvc2UtZXh0cmFjdG9yLnNlcnZpY2UudHMiXSwibmFtZXMiOltdLCJtYXBwaW5ncyI6IkFBQUEsT0FBTyxFQUFFLFlBQVksRUFBRSxVQUFVLEVBQUUsTUFBTSxlQUFlLENBQUM7QUFDekQsT0FBTyxFQUNMLGdCQUFnQixFQUNoQixRQUFRLEVBR1IsZ0JBQWdCLEVBQ2hCLGNBQWMsRUFDZCxtQkFBbUIsRUFDbkIsb0JBQW9CLEdBRXJCLE1BQU0scUJBQXFCLENBQUM7QUFDN0IsT0FBTyxFQUFFLGNBQWMsRUFBRSxhQUFhLEVBQUUsSUFBSSxFQUFFLE1BQU0sMEJBQTBCLENBQUM7O0FBRS9FOzs7O0dBSUc7QUFFSCxNQUFNLE9BQU8sb0JBQW9CO0lBYy9CO1FBYk8sMEJBQXFCLEdBR3ZCLElBQUksWUFBWSxFQUFFLENBQUM7UUFXdEIsSUFBSSxDQUFDLElBQUksRUFBRSxDQUFDO0lBQ2QsQ0FBQztJQUVNLHlCQUF5QjtRQUM5QixJQUFJLENBQUMsSUFBSSxDQUFDLHdCQUF3QjtZQUFFLE9BQU87UUFDM0MsT0FBTyxJQUFJLENBQUMsd0JBQXdCLENBQUMsYUFBYSxFQUFFLENBQUM7SUFDdkQsQ0FBQztJQUVNLHlCQUF5QjtRQUM5QixJQUFJLENBQUMsSUFBSSxDQUFDLHdCQUF3QjtZQUFFLE9BQU87UUFDM0MsT0FBTyxJQUFJLENBQUMsd0JBQXdCLENBQUMsYUFBYSxFQUFFLENBQUM7SUFDdkQsQ0FBQztJQUVNLEtBQUssQ0FBQyxZQUFZLENBQUMsWUFBOEI7UUFDdEQsSUFBSSxDQUFDLElBQUksQ0FBQyxRQUFRO1lBQUUsT0FBTztRQUUzQixJQUFJLElBQUksQ0FBQyx3QkFBd0IsRUFBRTtZQUNqQyxJQUFJLENBQUMsd0JBQXdCLENBQUMsS0FBSyxHQUFH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Q0FBQztnQkFDdEQsR0FBRyxDQUFDLE1BQU0sRUFBRSxDQUFDO2FBQ2Q7U0FDRjtJQUNILENBQUM7SUFFTyxjQUFjLENBQ3BCLFNBQWlDLEVBQ2pDLFFBQWtCO1FBRWxCLEtBQUssTUFBTSxPQUFPLElBQUksUUFBUSxFQUFFO1lBQzlCLE9BQU8sU0FBUyxDQUFDLE9BQU8sQ0FBQyxDQUFDO1NBQzNCO0lBQ0gsQ0FBQztJQUVPLGtCQUFrQixDQUFDLFNBQWdCLEVBQUUsS0FBYSxFQUFFLE1BQWM7UUFDeEUsTUFBTSxrQkFBa0IsR0FBRyxTQUFTLENBQUMsR0FBRyxDQUFDLENBQUMsUUFBUSxFQUFFLEVBQUUsQ0FBQyxRQUFRLENBQUMsQ0FBQyxHQUFHLEtBQUssQ0FBQyxDQUFDO1FBQzNFLE1BQU0sa0JBQWtCLEdBQUcsU0FBUyxDQUFDLEdBQUcsQ0FBQyxDQUFDLFFBQVEsRUFBRSxFQUFFLENBQUMsUUFBUSxDQUFDLENBQUMsR0FBRyxNQUFNLENBQUMsQ0FBQztRQUM1RSxNQUFNLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsa0JBQWtCLENBQUMsQ0FBQztRQUM3QyxNQUFNLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsa0JBQWtCLENBQUMsQ0FBQztRQUM3QyxNQUFNLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsa0JBQWtCLENBQUMsQ0FBQztRQUM3QyxNQUFNLElBQUksR0FBRyxJQUFJLENBQUMsR0FBRyxDQUFDLEdBQUcsa0JBQWtCLENBQUMsQ0FBQztRQUM3QyxPQUFPLENBQUMsSUFBSSxFQUFFLElBQUksRUFBRSxJQUFJLEdBQUcsSUFBSSxFQUFFLElBQUksR0FBRyxJQUFJLENBQUMsQ0FBQztJQUNoRCxDQUFDOztpSEEzU1Usb0JBQW9CO3FIQUFwQixvQkFBb0I7MkZBQXBCLG9CQUFvQjtrQkFEaEMsVUFBVSIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IEV2ZW50RW1pdHRlciwgSW5qZWN0YWJsZSB9IGZyb20gJ0Bhbmd1bGFyL2NvcmUnO1xuaW1wb3J0IHtcbiAgSEFORF9DT05ORUNUSU9OUyxcbiAgSG9saXN0aWMsXG4gIE5vcm1hbGl6ZWRMYW5kbWFyayxcbiAgTm9ybWFsaXplZExhbmRtYXJrTGlzdCxcbiAgUE9TRV9DT05ORUNUSU9OUyxcbiAgUE9TRV9MQU5ETUFSS1MsXG4gIFBPU0VfTEFORE1BUktTX0xFRlQsXG4gIFBPU0VfTEFORE1BUktTX1JJR0hULFxuICBSZXN1bHRzLFxufSBmcm9tICdAbWVkaWFwaXBlL2hvbGlzdGljJztcbmltcG9ydCB7IGRyYXdDb25uZWN0b3JzLCBkcmF3TGFuZG1hcmtzLCBsZXJwIH0gZnJvbSAnQG1lZGlhcGlwZS9kcmF3aW5nX3V0aWxzJztcblxuLyoqXG4gKiBNZWRpYVBpcGUg44KS55So44GE44Gm5YuV55S744GL44KJ44Od44O844K644KS5oq95Ye644GZ44KL44Gf44KB44Gu44K144O844OT44K5XG4gKlxuICog4oC7IOOCt+ODs+OCsOODq+ODiOODs+OBquOCteODvOODk+OCueOBp+OBr+OBquOBhOOBn+OCgeOAgUNvbXBvbmVudCDjgacgcHJvdmlkZXJzIOOBq+aMh+WumuOBl+OBpuS9v+eUqOOBmeOCi+OBk+OBqOOCkuaDs+WumlxuICovXG5ASW5qZWN0YWJsZSgpXG5leHBvcnQgY2xhc3MgUG9zZUV4dHJhY3RvclNlcnZpY2Uge1xuICBwdWJsaWMgb25SZXN1bHRzRXZlbnRFbWl0dGVyOiBFdmVudEVtaXR0ZXI8e1xuICAgIG1wUmVzdWx0czogUmVzdWx0cztcbiAgICBwb3NlUHJldmlld0ltYWdlRGF0YVVybDogc3RyaW5nO1xuICB9PiA9IG5ldyBFdmVudEVtaXR0ZXIoKTtcblxuICBwcml2YXRlIGhvbGlzdGljPzogSG9saXN0aWM7XG5cbiAgcHJpdmF0ZSBwb3NlUHJldmlld0NhbnZhc0VsZW1lbnQ/OiBIVE1MQ2FudmFzRWxlbWVudDtcbiAgcHJpdmF0ZSBwb3NlUHJldmlld0NhbnZhc0NvbnRleHQ/OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG5cbiAgcHJpdmF0ZSBoYW5kUHJldmlld0NhbnZhc0VsZW1lbnQ/OiBIVE1MQ2FudmFzRWxlbWVudDtcbiAgcHJpdmF0ZSBoYW5kUHJldmlld0NhbnZhc0NvbnRleHQ/OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQ7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy5pbml0KCk7XG4gIH1cblxuICBwdWJsaWMgZ2V0UG9zZVByZXZpZXdNZWRpYVN0cmVhbSgpOiBNZWRpYVN0cmVhbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKCF0aGlzLnBvc2VQcmV2aWV3Q2FudmFzRWxlbWVudCkgcmV0dXJuO1xuICAgIHJldHVybiB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzRWxlbWVudC5jYXB0dXJlU3RyZWFtKCk7XG4gIH1cblxuICBwdWJsaWMgZ2V0SGFuZFByZXZpZXdNZWRpYVN0cmVhbSgpOiBNZWRpYVN0cmVhbSB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKCF0aGlzLmhhbmRQcmV2aWV3Q2FudmFzRWxlbWVudCkgcmV0dXJuO1xuICAgIHJldHVybiB0aGlzLmhhbmRQcmV2aWV3Q2FudmFzRWxlbWVudC5jYXB0dXJlU3RyZWFtKCk7XG4gIH1cblxuICBwdWJsaWMgYXN5bmMgb25WaWRlb0ZyYW1lKHZpZGVvRWxlbWVudDogSFRNTFZpZGVvRWxlbWVudCkge1xuICAgIGlmICghdGhpcy5ob2xpc3RpYykgcmV0dXJuO1xuXG4gICAgaWYgKHRoaXMucG9zZVByZXZpZXdDYW52YXNFbGVtZW50KSB7XG4gICAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzRWxlbWVudC53aWR0aCA9IHZpZGVvRWxlbWVudC52aWRlb1dpZHRoO1xuICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0VsZW1lbnQuaGVpZ2h0ID0gdmlkZW9FbGVtZW50LnZpZGVvSGVpZ2h0O1xuICAgIH1cblxuICAgIGlmICh0aGlzLmhhbmRQcmV2aWV3Q2FudmFzRWxlbWVudCkge1xuICAgICAgdGhpcy5oYW5kUHJldmlld0NhbnZhc0VsZW1lbnQud2lkdGggPSB2aWRlb0VsZW1lbnQudmlkZW9XaWR0aDtcbiAgICAgIHRoaXMuaGFuZFByZXZpZXdDYW52YXNFbGVtZW50LmhlaWdodCA9IHZpZGVvRWxlbWVudC52aWRlb0hlaWdodDtcbiAgICB9XG5cbiAgICBhd2FpdCB0aGlzLmhvbGlzdGljLnNlbmQoeyBpbWFnZTogdmlkZW9FbGVtZW50IH0pO1xuICB9XG5cbiAgcHJpdmF0ZSBpbml0KCkge1xuICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNFbGVtZW50ID0gZG9jdW1lbnQuY3JlYXRlRWxlbWVudCgnY2FudmFzJyk7XG4gICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0NvbnRleHQgPVxuICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0VsZW1lbnQuZ2V0Q29udGV4dCgnMmQnKSB8fCB1bmRlZmluZWQ7XG5cbiAgICB0aGlzLmhhbmRQcmV2aWV3Q2FudmFzRWxlbWVudCA9IGRvY3VtZW50LmNyZWF0ZUVsZW1lbnQoJ2NhbnZhcycpO1xuICAgIHRoaXMuaGFuZFByZXZpZXdDYW52YXNDb250ZXh0ID1cbiAgICAgIHRoaXMuaGFuZFByZXZpZXdDYW52YXNFbGVtZW50LmdldENvbnRleHQoJzJkJykgfHwgdW5kZWZpbmVkO1xuXG4gICAgdGhpcy5ob2xpc3RpYyA9IG5ldyBIb2xpc3RpYyh7XG4gICAgICBsb2NhdGVGaWxlOiAoZmlsZSkgPT4ge1xuICAgICAgICByZXR1cm4gYGh0dHBzOi8vY2RuLmpzZGVsaXZyLm5ldC9ucG0vQG1lZGlhcGlwZS9ob2xpc3RpYy8ke2ZpbGV9YDtcbiAgICAgIH0sXG4gICAgfSk7XG5cbiAgICB0aGlzLmhvbGlzdGljLnNldE9wdGlvbnMoe1xuICAgICAgc2VsZmllTW9kZTogZmFsc2UsXG4gICAgICBtb2RlbENvbXBsZXhpdHk6IDEsXG4gICAgICBzbW9vdGhMYW5kbWFya3M6IHRydWUsXG4gICAgICBlbmFibGVTZWdtZW50YXRpb246IGZhbHNlLFxuICAgICAgc21vb3RoU2VnbWVudGF0aW9uOiB0cnVlLFxuICAgICAgbWluRGV0ZWN0aW9uQ29uZmlkZW5jZTogMC42LFxuICAgICAgbWluVHJhY2tpbmdDb25maWRlbmNlOiAwLjYsXG4gICAgfSk7XG5cbiAgICB0aGlzLmhvbGlzdGljLm9uUmVzdWx0cygocmVzdWx0czogUmVzdWx0cykgPT4ge1xuICAgICAgdGhpcy5vblJlc3VsdHMocmVzdWx0cyk7XG4gICAgfSk7XG4gIH1cblxuICBwcml2YXRlIG9uUmVzdWx0cyhyZXN1bHRzOiBSZXN1bHRzKSB7XG4gICAgaWYgKFxuICAgICAgIXRoaXMucG9zZVByZXZpZXdDYW52YXNFbGVtZW50IHx8XG4gICAgICAhdGhpcy5wb3NlUHJldmlld0NhbnZhc0NvbnRleHQgfHxcbiAgICAgICF0aGlzLmhvbGlzdGljXG4gICAgKVxuICAgICAgcmV0dXJuO1xuXG4gICAgLy8g5o+P55S755So44Gr5LiN5b+F6KaB44Gq44Op44Oz44OJ44Oe44O844Kv44KS6Zmk5Y67XG4gICAgbGV0IHBvc2VMYW5kbWFya3M6IE5vcm1hbGl6ZWRMYW5kbWFya0xpc3QgPSBbXTtcbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzKSB7XG4gICAgICBwb3NlTGFuZG1hcmtzID0gSlNPTi5wYXJzZShcbiAgICAgICAgSlNPTi5zdHJpbmdpZnkocmVzdWx0cy5wb3NlTGFuZG1hcmtzKVxuICAgICAgKSBhcyBOb3JtYWxpemVkTGFuZG1hcmtMaXN0O1xuICAgICAgdGhpcy5yZW1vdmVFbGVtZW50cyhcbiAgICAgICAgcG9zZUxhbmRtYXJrcyxcbiAgICAgICAgWzAsIDEsIDIsIDMsIDQsIDUsIDYsIDcsIDgsIDksIDEwLCAxNSwgMTYsIDE3LCAxOCwgMTksIDIwLCAyMSwgMjJdXG4gICAgICApO1xuICAgIH1cblxuICAgIC8vIOOCreODo+ODs+ODkOOCueOCkuWhl+OCiuOBpOOBtuOBl1xuICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNDb250ZXh0LnNhdmUoKTtcbiAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dC5jbGVhclJlY3QoXG4gICAgICAwLFxuICAgICAgMCxcbiAgICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNFbGVtZW50LndpZHRoLFxuICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0VsZW1lbnQuaGVpZ2h0XG4gICAgKTtcblxuICAgIC8vIOaknOWHuuOBq+S9v+eUqOOBl+OBn+ODleODrOODvOODoOeUu+WDj+OCkuaPj+eUu1xuICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNDb250ZXh0LmRyYXdJbWFnZShcbiAgICAgIHJlc3VsdHMuaW1hZ2UsXG4gICAgICAwLFxuICAgICAgMCxcbiAgICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNFbGVtZW50LndpZHRoLFxuICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0VsZW1lbnQuaGVpZ2h0XG4gICAgKTtcblxuICAgIC8vIOiCmOOBqOaJi+OCkuOBpOOBquOBkOe3muOCkuaPj+eUu1xuICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNDb250ZXh0LmxpbmVXaWR0aCA9IDU7XG4gICAgaWYgKHBvc2VMYW5kbWFya3MpIHtcbiAgICAgIGlmIChyZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcykge1xuICAgICAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dC5zdHJva2VTdHlsZSA9ICd3aGl0ZSc7XG4gICAgICAgIHRoaXMuY29ubmVjdCh0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dCwgW1xuICAgICAgICAgIFtcbiAgICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLFxuICAgICAgICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NbMF0sXG4gICAgICAgICAgXSxcbiAgICAgICAgXSk7XG4gICAgICB9XG4gICAgICBpZiAocmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcykge1xuICAgICAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dC5zdHJva2VTdHlsZSA9ICd3aGl0ZSc7XG4gICAgICAgIHRoaXMuY29ubmVjdCh0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dCwgW1xuICAgICAgICAgIFtcbiAgICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10sXG4gICAgICAgICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzWzBdLFxuICAgICAgICAgIF0sXG4gICAgICAgIF0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOODneODvOOCuuOBruODl+ODrOODk+ODpeODvOOCkuaPj+eUu1xuICAgIGlmIChwb3NlTGFuZG1hcmtzKSB7XG4gICAgICBkcmF3Q29ubmVjdG9ycyhcbiAgICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0NvbnRleHQsXG4gICAgICAgIHBvc2VMYW5kbWFya3MsXG4gICAgICAgIFBPU0VfQ09OTkVDVElPTlMsXG4gICAgICAgIHsgY29sb3I6ICd3aGl0ZScgfVxuICAgICAgKTtcbiAgICAgIGRyYXdMYW5kbWFya3MoXG4gICAgICAgIHRoaXMucG9zZVByZXZpZXdDYW52YXNDb250ZXh0LFxuICAgICAgICBPYmplY3QudmFsdWVzKFBPU0VfTEFORE1BUktTX0xFRlQpLm1hcCgoaW5kZXgpID0+IHBvc2VMYW5kbWFya3NbaW5kZXhdKSxcbiAgICAgICAgeyB2aXNpYmlsaXR5TWluOiAwLjY1LCBjb2xvcjogJ3doaXRlJywgZmlsbENvbG9yOiAncmdiKDI1NSwxMzgsMCknIH1cbiAgICAgICk7XG4gICAgICBkcmF3TGFuZG1hcmtzKFxuICAgICAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dCxcbiAgICAgICAgT2JqZWN0LnZhbHVlcyhQT1NFX0xBTkRNQVJLU19SSUdIVCkubWFwKFxuICAgICAgICAgIChpbmRleCkgPT4gcG9zZUxhbmRtYXJrc1tpbmRleF1cbiAgICAgICAgKSxcbiAgICAgICAgeyB2aXNpYmlsaXR5TWluOiAwLjY1LCBjb2xvcjogJ3doaXRlJywgZmlsbENvbG9yOiAncmdiKDAsMjE3LDIzMSknIH1cbiAgICAgICk7XG4gICAgfVxuXG4gICAgLy8g5omL44Gu44OX44Os44OT44Ol44O844KS5o+P55S7XG4gICAgZHJhd0Nvbm5lY3RvcnMoXG4gICAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dCxcbiAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzLFxuICAgICAgSEFORF9DT05ORUNUSU9OUyxcbiAgICAgIHsgY29sb3I6ICd3aGl0ZScgfVxuICAgICk7XG4gICAgZHJhd0xhbmRtYXJrcyh0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dCwgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MsIHtcbiAgICAgIGNvbG9yOiAnd2hpdGUnLFxuICAgICAgZmlsbENvbG9yOiAncmdiKDAsMjE3LDIzMSknLFxuICAgICAgbGluZVdpZHRoOiAyLFxuICAgICAgcmFkaXVzOiAoZGF0YTogYW55KSA9PiB7XG4gICAgICAgIHJldHVybiBsZXJwKGRhdGEuZnJvbS56LCAtMC4xNSwgMC4xLCAxMCwgMSk7XG4gICAgICB9LFxuICAgIH0pO1xuICAgIGRyYXdDb25uZWN0b3JzKFxuICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0NvbnRleHQsXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgSEFORF9DT05ORUNUSU9OUyxcbiAgICAgIHsgY29sb3I6ICd3aGl0ZScgfVxuICAgICk7XG4gICAgZHJhd0xhbmRtYXJrcyh0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dCwgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcywge1xuICAgICAgY29sb3I6ICd3aGl0ZScsXG4gICAgICBmaWxsQ29sb3I6ICdyZ2IoMjU1LDEzOCwwKScsXG4gICAgICBsaW5lV2lkdGg6IDIsXG4gICAgICByYWRpdXM6IChkYXRhOiBhbnkpID0+IHtcbiAgICAgICAgcmV0dXJuIGxlcnAoZGF0YS5mcm9tLnosIC0wLjE1LCAwLjEsIDEwLCAxKTtcbiAgICAgIH0sXG4gICAgfSk7XG5cbiAgICAvLyDmiYvjga7poJjln5/jga7jgb/jga7jg5fjg6zjg5Pjg6Xjg7zjgpLnlJ/miJBcbiAgICBpZiAodGhpcy5oYW5kUHJldmlld0NhbnZhc0NvbnRleHQgJiYgdGhpcy5oYW5kUHJldmlld0NhbnZhc0VsZW1lbnQpIHtcbiAgICAgIGNvbnN0IEhBTkRfUFJFVklFV19aT09NID0gMztcbiAgICAgIGNvbnN0IGhhbmRQcmV2aWV3QmFzZVkgPSB0aGlzLmhhbmRQcmV2aWV3Q2FudmFzRWxlbWVudC5oZWlnaHQgLyAyO1xuXG4gICAgICB0aGlzLmhhbmRQcmV2aWV3Q2FudmFzQ29udGV4dC5jbGVhclJlY3QoXG4gICAgICAgIDAsXG4gICAgICAgIDAsXG4gICAgICAgIHRoaXMuaGFuZFByZXZpZXdDYW52YXNFbGVtZW50LndpZHRoLFxuICAgICAgICB0aGlzLmhhbmRQcmV2aWV3Q2FudmFzRWxlbWVudC5oZWlnaHRcbiAgICAgICk7XG5cbiAgICAgIGlmIChyZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcykge1xuICAgICAgICBjb25zdCByZWN0ID0gdGhpcy5nZXRSZWN0QnlMYW5kbWFya3MoXG4gICAgICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3MsXG4gICAgICAgICAgcmVzdWx0cy5pbWFnZS53aWR0aCxcbiAgICAgICAgICByZXN1bHRzLmltYWdlLmhlaWdodFxuICAgICAgICApO1xuXG4gICAgICAgIGxldCBoYW5kUHJldmlld1ggPSAwO1xuICAgICAgICBsZXQgaGFuZFByZXZpZXdZID0gaGFuZFByZXZpZXdCYXNlWSAtIChyZWN0WzNdICogSEFORF9QUkVWSUVXX1pPT00pIC8gMjtcblxuICAgICAgICB0aGlzLmhhbmRQcmV2aWV3Q2FudmFzQ29udGV4dC5kcmF3SW1hZ2UoXG4gICAgICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0VsZW1lbnQsXG4gICAgICAgICAgcmVjdFswXSAtIDEwLFxuICAgICAgICAgIHJlY3RbMV0gLSAxMCxcbiAgICAgICAgICByZWN0WzJdICsgMTAsXG4gICAgICAgICAgcmVjdFszXSArIDEwLFxuICAgICAgICAgIGhhbmRQcmV2aWV3WCxcbiAgICAgICAgICBoYW5kUHJldmlld1ksXG4gICAgICAgICAgcmVjdFsyXSAqIEhBTkRfUFJFVklFV19aT09NLFxuICAgICAgICAgIHJlY3RbM10gKiBIQU5EX1BSRVZJRVdfWk9PTVxuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBpZiAocmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcykge1xuICAgICAgICBjb25zdCByZWN0ID0gdGhpcy5nZXRSZWN0QnlMYW5kbWFya3MoXG4gICAgICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgICAgICByZXN1bHRzLmltYWdlLndpZHRoLFxuICAgICAgICAgIHJlc3VsdHMuaW1hZ2UuaGVpZ2h0XG4gICAgICAgICk7XG5cbiAgICAgICAgbGV0IGhhbmRQcmV2aWV3WCA9XG4gICAgICAgICAgdGhpcy5oYW5kUHJldmlld0NhbnZhc0VsZW1lbnQud2lkdGggLSByZWN0WzJdICogSEFORF9QUkVWSUVXX1pPT007XG4gICAgICAgIGxldCBoYW5kUHJldmlld1kgPSBoYW5kUHJldmlld0Jhc2VZIC0gKHJlY3RbM10gKiBIQU5EX1BSRVZJRVdfWk9PTSkgLyAyO1xuXG4gICAgICAgIHRoaXMuaGFuZFByZXZpZXdDYW52YXNDb250ZXh0LmRyYXdJbWFnZShcbiAgICAgICAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzRWxlbWVudCxcbiAgICAgICAgICByZWN0WzBdIC0gMTAsXG4gICAgICAgICAgcmVjdFsxXSAtIDEwLFxuICAgICAgICAgIHJlY3RbMl0gKyAxMCxcbiAgICAgICAgICByZWN0WzNdICsgMTAsXG4gICAgICAgICAgaGFuZFByZXZpZXdYLFxuICAgICAgICAgIGhhbmRQcmV2aWV3WSxcbiAgICAgICAgICByZWN0WzJdICogSEFORF9QUkVWSUVXX1pPT00sXG4gICAgICAgICAgcmVjdFszXSAqIEhBTkRfUFJFVklFV19aT09NXG4gICAgICAgICk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8g44Kk44OZ44Oz44OI44KS6YCB5Ye6XG4gICAgdGhpcy5vblJlc3VsdHNFdmVudEVtaXR0ZXIuZW1pdCh7XG4gICAgICBtcFJlc3VsdHM6IHJlc3VsdHMsXG4gICAgICBwb3NlUHJldmlld0ltYWdlRGF0YVVybDpcbiAgICAgICAgdGhpcy5wb3NlUHJldmlld0NhbnZhc0VsZW1lbnQudG9EYXRhVVJMKCdpbWFnZS9qcGVnJyksXG4gICAgfSk7XG5cbiAgICAvLyDlrozkuoZcbiAgICB0aGlzLnBvc2VQcmV2aWV3Q2FudmFzQ29udGV4dC5yZXN0b3JlKCk7XG4gIH1cblxuICBwcml2YXRlIGNvbm5lY3QoXG4gICAgY3R4OiBDYW52YXNSZW5kZXJpbmdDb250ZXh0MkQsXG4gICAgY29ubmVjdG9yczogQXJyYXk8W05vcm1hbGl6ZWRMYW5kbWFyaywgTm9ybWFsaXplZExhbmRtYXJrXT5cbiAgKSB7XG4gICAgY29uc3QgY2FudmFzID0gY3R4LmNhbnZhcztcbiAgICBmb3IgKGNvbnN0IGNvbm5lY3RvciBvZiBjb25uZWN0b3JzKSB7XG4gICAgICBjb25zdCBmcm9tID0gY29ubmVjdG9yWzBdO1xuICAgICAgY29uc3QgdG8gPSBjb25uZWN0b3JbMV07XG4gICAgICBpZiAoZnJvbSAmJiB0bykge1xuICAgICAgICBpZiAoXG4gICAgICAgICAgZnJvbS52aXNpYmlsaXR5ICYmXG4gICAgICAgICAgdG8udmlzaWJpbGl0eSAmJlxuICAgICAgICAgIChmcm9tLnZpc2liaWxpdHkgPCAwLjEgfHwgdG8udmlzaWJpbGl0eSA8IDAuMSlcbiAgICAgICAgKSB7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgY3R4LmJlZ2luUGF0aCgpO1xuICAgICAgICBjdHgubW92ZVRvKGZyb20ueCAqIGNhbnZhcy53aWR0aCwgZnJvbS55ICogY2FudmFzLmhlaWdodCk7XG4gICAgICAgIGN0eC5saW5lVG8odG8ueCAqIGNhbnZhcy53aWR0aCwgdG8ueSAqIGNhbnZhcy5oZWlnaHQpO1xuICAgICAgICBjdHguc3Ryb2tlKCk7XG4gICAgICB9XG4gICAgfVxuICB9XG5cbiAgcHJpdmF0ZSByZW1vdmVFbGVtZW50cyhcbiAgICBsYW5kbWFya3M6IE5vcm1hbGl6ZWRMYW5kbWFya0xpc3QsXG4gICAgZWxlbWVudHM6IG51bWJlcltdXG4gICkge1xuICAgIGZvciAoY29uc3QgZWxlbWVudCBvZiBlbGVtZW50cykge1xuICAgICAgZGVsZXRlIGxhbmRtYXJrc1tlbGVtZW50XTtcbiAgICB9XG4gIH1cblxuICBwcml2YXRlIGdldFJlY3RCeUxhbmRtYXJrcyhsYW5kbWFya3M6IGFueVtdLCB3aWR0aDogbnVtYmVyLCBoZWlnaHQ6IG51bWJlcikge1xuICAgIGNvbnN0IGxlZnRIYW5kTGFuZG1hcmtzWCA9IGxhbmRtYXJrcy5tYXAoKGxhbmRtYXJrKSA9PiBsYW5kbWFyay54ICogd2lkdGgpO1xuICAgIGNvbnN0IGxlZnRIYW5kTGFuZG1hcmtzWSA9IGxhbmRtYXJrcy5tYXAoKGxhbmRtYXJrKSA9PiBsYW5kbWFyay55ICogaGVpZ2h0KTtcbiAgICBjb25zdCBtaW5YID0gTWF0aC5taW4oLi4ubGVmdEhhbmRMYW5kbWFya3NYKTtcbiAgICBjb25zdCBtYXhYID0gTWF0aC5tYXgoLi4ubGVmdEhhbmRMYW5kbWFya3NYKTtcbiAgICBjb25zdCBtaW5ZID0gTWF0aC5taW4oLi4ubGVmdEhhbmRMYW5kbWFya3NZKTtcbiAgICBjb25zdCBtYXhZID0gTWF0aC5tYXgoLi4ubGVmdEhhbmRMYW5kbWFya3NZKTtcbiAgICByZXR1cm4gW21pblgsIG1pblksIG1heFggLSBtaW5YLCBtYXhZIC0gbWluWV07XG4gIH1cbn1cbiJdfQ==