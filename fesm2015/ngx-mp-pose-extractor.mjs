import * as i0 from '@angular/core';
import { Injectable, Component, NgModule, EventEmitter } from '@angular/core';
import { __awaiter } from 'tslib';
import { POSE_LANDMARKS, Holistic, POSE_CONNECTIONS, POSE_LANDMARKS_LEFT, POSE_LANDMARKS_RIGHT, HAND_CONNECTIONS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import cosSimilarity from 'cos-similarity';
import { drawConnectors, drawLandmarks, lerp } from '@mediapipe/drawing_utils';

class NgxMpPoseExtractorService {
    constructor() { }
}
NgxMpPoseExtractorService.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorService, deps: [], target: i0.ɵɵFactoryTarget.Injectable });
NgxMpPoseExtractorService.ɵprov = i0.ɵɵngDeclareInjectable({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorService, providedIn: 'root' });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorService, decorators: [{
            type: Injectable,
            args: [{
                    providedIn: 'root'
                }]
        }], ctorParameters: function () { return []; } });

class NgxMpPoseExtractorComponent {
}
NgxMpPoseExtractorComponent.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorComponent, deps: [], target: i0.ɵɵFactoryTarget.Component });
NgxMpPoseExtractorComponent.ɵcmp = i0.ɵɵngDeclareComponent({ minVersion: "14.0.0", version: "15.0.4", type: NgxMpPoseExtractorComponent, selector: "lib-ngx-mp-pose-extractor", ngImport: i0, template: `
    <p>
      ngx-mp-pose-extractor works!
    </p>
  `, isInline: true });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorComponent, decorators: [{
            type: Component,
            args: [{ selector: 'lib-ngx-mp-pose-extractor', template: `
    <p>
      ngx-mp-pose-extractor works!
    </p>
  ` }]
        }] });

class NgxMpPoseExtractorModule {
}
NgxMpPoseExtractorModule.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule, deps: [], target: i0.ɵɵFactoryTarget.NgModule });
NgxMpPoseExtractorModule.ɵmod = i0.ɵɵngDeclareNgModule({ minVersion: "14.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule, declarations: [NgxMpPoseExtractorComponent], exports: [NgxMpPoseExtractorComponent] });
NgxMpPoseExtractorModule.ɵinj = i0.ɵɵngDeclareInjector({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: NgxMpPoseExtractorModule, decorators: [{
            type: NgModule,
            args: [{
                    declarations: [NgxMpPoseExtractorComponent],
                    imports: [],
                    exports: [NgxMpPoseExtractorComponent],
                }]
        }] });

class Pose {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        this.videoMetadata = {
            name: '',
            width: 0,
            height: 0,
            duration: 0,
        };
    }
    getVideoName() {
        return this.videoMetadata.name;
    }
    setVideoName(videoName) {
        this.videoMetadata.name = videoName;
    }
    setVideoMetaData(width, height, duration) {
        this.videoMetadata.width = width;
        this.videoMetadata.height = height;
        this.videoMetadata.duration = duration;
    }
    getNumberOfPoses() {
        if (this.poses === undefined)
            return -1;
        return this.poses.length;
    }
    getPoses() {
        if (this.poses === undefined)
            return [];
        return this.poses;
    }
    pushPose(videoTimeMiliseconds, frameImageJpegDataUrl, poseImageJpegDataUrl, videoWidth, videoHeight, videoDuration, results) {
        this.setVideoMetaData(videoWidth, videoHeight, videoDuration);
        if (results.poseLandmarks === undefined)
            return;
        const poseLandmarksWithWorldCoordinate = results.ea
            ? results.ea
            : [];
        if (poseLandmarksWithWorldCoordinate.length === 0) {
            console.warn(`[Pose] pushPose - Could not get the pose with the world coordinate`, results);
            return;
        }
        const poseVector = Pose.getPoseVector(poseLandmarksWithWorldCoordinate);
        if (!poseVector) {
            console.warn(`[Pose] pushPose - Could not get the pose vector`, poseLandmarksWithWorldCoordinate);
            return;
        }
        const pose = {
            t: videoTimeMiliseconds,
            pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
                return [landmark.x, landmark.y, landmark.z, landmark.visibility];
            }),
            vectors: poseVector,
            frameImageDataUrl: frameImageJpegDataUrl,
            poseImageDataUrl: poseImageJpegDataUrl,
        };
        if (1 <= this.poses.length) {
            const lastPose = this.poses[this.poses.length - 1];
            if (Pose.isSimilarPose(lastPose.vectors, pose.vectors)) {
                return;
            }
        }
        this.poses.push(pose);
    }
    finalize() {
        if (Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
            // 全ポーズを走査して、類似するポーズを削除する
            const newPoses = [];
            for (const poseA of this.poses) {
                let isDuplicated = false;
                for (const poseB of newPoses) {
                    if (Pose.isSimilarPose(poseA.vectors, poseB.vectors)) {
                        isDuplicated = true;
                        break;
                    }
                }
                if (isDuplicated)
                    continue;
                newPoses.push(poseA);
            }
            console.info(`[Pose] getJson - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`);
            this.poses = newPoses;
        }
        this.isFinalized = true;
    }
    getSimilarPoses(results) {
        const poseVector = Pose.getPoseVector(results.ea);
        if (!poseVector)
            throw 'Could not get the pose vector';
        return this.poses.filter((p) => Pose.isSimilarPose(p.vectors, poseVector));
    }
    static getPoseVector(poseLandmarks) {
        return {
            rightWristToRightElbow: [
                poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].x -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].x,
                poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].y -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].y,
                poseLandmarks[POSE_LANDMARKS.RIGHT_WRIST].z -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].z,
            ],
            rightElbowToRightShoulder: [
                poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].x -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].x,
                poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].y -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].y,
                poseLandmarks[POSE_LANDMARKS.RIGHT_ELBOW].z -
                    poseLandmarks[POSE_LANDMARKS.RIGHT_SHOULDER].z,
            ],
            leftWristToLeftElbow: [
                poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].x -
                    poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].x,
                poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].y -
                    poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].y,
                poseLandmarks[POSE_LANDMARKS.LEFT_WRIST].z -
                    poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].z,
            ],
            leftElbowToLeftShoulder: [
                poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].x -
                    poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].x,
                poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].y -
                    poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].y,
                poseLandmarks[POSE_LANDMARKS.LEFT_ELBOW].z -
                    poseLandmarks[POSE_LANDMARKS.LEFT_SHOULDER].z,
            ],
        };
    }
    static isSimilarPose(poseVectorA, poseVectorB, threshold = 0.9) {
        const cosSimilarities = {
            leftWristToLeftElbow: cosSimilarity(poseVectorA.leftWristToLeftElbow, poseVectorB.leftWristToLeftElbow),
            leftElbowToLeftShoulder: cosSimilarity(poseVectorA.leftElbowToLeftShoulder, poseVectorB.leftElbowToLeftShoulder),
            rightWristToRightElbow: cosSimilarity(poseVectorA.rightWristToRightElbow, poseVectorB.rightWristToRightElbow),
            rightElbowToRightShoulder: cosSimilarity(poseVectorA.rightElbowToRightShoulder, poseVectorB.rightElbowToRightShoulder),
        };
        let isSimilar = false;
        const cosSimilaritiesSum = Object.values(cosSimilarities).reduce((sum, value) => sum + value, 0);
        if (cosSimilaritiesSum >= threshold * Object.keys(cosSimilarities).length)
            isSimilar = true;
        console.log(`[Pose] isSimilarPose`, isSimilar, cosSimilarities);
        return isSimilar;
    }
    getZip() {
        return __awaiter(this, void 0, void 0, function* () {
            const jsZip = new JSZip();
            jsZip.file('poses.json', this.getJson());
            for (const pose of this.poses) {
                if (pose.frameImageDataUrl) {
                    try {
                        const index = pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                        const base64 = pose.frameImageDataUrl.substring(index);
                        jsZip.file(`frame-${pose.t}.jpg`, base64, { base64: true });
                    }
                    catch (error) {
                        console.warn(`[PoseExporterService] push - Could not push frame image`, error);
                        throw error;
                    }
                }
                if (pose.poseImageDataUrl) {
                    try {
                        const index = pose.poseImageDataUrl.indexOf('base64,') + 'base64,'.length;
                        const base64 = pose.poseImageDataUrl.substring(index);
                        jsZip.file(`pose-${pose.t}.jpg`, base64, { base64: true });
                    }
                    catch (error) {
                        console.warn(`[PoseExporterService] push - Could not push frame image`, error);
                        throw error;
                    }
                }
            }
            return yield jsZip.generateAsync({ type: 'blob' });
        });
    }
    getJson() {
        if (this.videoMetadata === undefined || this.poses === undefined)
            return '{}';
        if (!this.isFinalized) {
            this.finalize();
        }
        let poseLandmarkMappings = [];
        for (const key of Object.keys(POSE_LANDMARKS)) {
            const index = POSE_LANDMARKS[key];
            poseLandmarkMappings[index] = key;
        }
        const json = {
            generator: 'mp-video-pose-extractor',
            version: 1,
            video: this.videoMetadata,
            poses: this.poses.map((pose) => {
                const poseVector = [];
                for (const key of Pose.POSE_VECTOR_MAPPINGS) {
                    poseVector.push(pose.vectors[key]);
                }
                return {
                    t: pose.t,
                    pose: pose.pose,
                    vectors: poseVector,
                };
            }),
            poseLandmarkMapppings: poseLandmarkMappings,
        };
        return JSON.stringify(json);
    }
    loadJson(json) {
        const parsedJson = typeof json === 'string' ? JSON.parse(json) : json;
        if (parsedJson.generator !== 'mp-video-pose-extractor') {
            throw '不正なファイル';
        }
        else if (parsedJson.version !== 1) {
            throw '未対応のバージョン';
        }
        this.videoMetadata = parsedJson.video;
        this.poses = parsedJson.poses.map((poseJsonItem) => {
            const poseVector = {};
            Pose.POSE_VECTOR_MAPPINGS.map((key, index) => {
                poseVector[key] = poseJsonItem.vectors[index];
            });
            return {
                t: poseJsonItem.t,
                pose: poseJsonItem.pose,
                vectors: poseVector,
                frameImageDataUrl: undefined,
            };
        });
    }
    loadZip(buffer, includeImages = true) {
        var _a, _b, _c;
        return __awaiter(this, void 0, void 0, function* () {
            const jsZip = new JSZip();
            const zip = yield jsZip.loadAsync(buffer, { base64: false });
            if (!zip)
                throw 'ZIPファイルを読み込めませんでした';
            const json = yield ((_a = zip.file('poses.json')) === null || _a === void 0 ? void 0 : _a.async('text'));
            if (json === undefined) {
                throw 'ZIPファイルに pose.json が含まれていません';
            }
            this.loadJson(json);
            if (includeImages) {
                for (const pose of this.poses) {
                    if (!pose.frameImageDataUrl) {
                        const frameImageFileName = `frame-${pose.t}.jpg`;
                        const imageBase64 = yield ((_b = zip
                            .file(frameImageFileName)) === null || _b === void 0 ? void 0 : _b.async('base64'));
                        if (imageBase64) {
                            pose.frameImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
                        }
                    }
                    if (!pose.poseImageDataUrl) {
                        const poseImageFileName = `pose-${pose.t}.jpg`;
                        const imageBase64 = yield ((_c = zip
                            .file(poseImageFileName)) === null || _c === void 0 ? void 0 : _c.async('base64'));
                        if (imageBase64) {
                            pose.poseImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
                        }
                    }
                }
            }
        });
    }
}
Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;
Pose.POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
];

/**
 * ポーズを管理するためのサービス
 */
class PoseComposerService {
    constructor() { }
    init(videoName) {
        const pose = new Pose();
        pose.setVideoName(videoName);
        return pose;
    }
    downloadAsJson(pose) {
        const blob = new Blob([pose.getJson()], {
            type: 'application/json',
        });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${pose.getVideoName()}-poses.json`;
        a.click();
    }
    downloadAsZip(pose) {
        return __awaiter(this, void 0, void 0, function* () {
            const content = yield pose.getZip();
            const url = window.URL.createObjectURL(content);
            const a = document.createElement('a');
            a.href = url;
            a.target = '_blank';
            a.download = `${pose.getVideoName()}-poses.zip`;
            a.click();
        });
    }
}
PoseComposerService.ɵfac = i0.ɵɵngDeclareFactory({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseComposerService, deps: [], target: i0.ɵɵFactoryTarget.Injectable });
PoseComposerService.ɵprov = i0.ɵɵngDeclareInjectable({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseComposerService, providedIn: 'root' });
i0.ɵɵngDeclareClassMetadata({ minVersion: "12.0.0", version: "15.0.4", ngImport: i0, type: PoseComposerService, decorators: [{
            type: Injectable,
            args: [{
                    providedIn: 'root',
                }]
        }], ctorParameters: function () { return []; } });

/**
 * MediaPipe を用いて動画からポーズを抽出するためのサービス
 *
 * ※ シングルトンなサービスではないため、Component で providers に指定して使用することを想定
 */
class PoseExtractorService {
    constructor() {
        this.onResultsEventEmitter = new EventEmitter();
        this.IMAGE_JPEG_QUALITY = 0.8;
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
    onVideoFrame(videoElement) {
        return __awaiter(this, void 0, void 0, function* () {
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
            yield this.holistic.send({ image: videoElement });
        });
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
        // 検出に使用したフレーム画像を保持
        const sourceImageDataUrl = this.posePreviewCanvasElement.toDataURL('image/jpeg', this.IMAGE_JPEG_QUALITY);
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
            sourceImageDataUrl: sourceImageDataUrl,
            posePreviewImageDataUrl: this.posePreviewCanvasElement.toDataURL('image/jpeg', this.IMAGE_JPEG_QUALITY),
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

/*
 * Public API Surface of ngx-mp-pose-extractor
 */

/**
 * Generated bundle index. Do not edit.
 */

export { NgxMpPoseExtractorComponent, NgxMpPoseExtractorModule, NgxMpPoseExtractorService, Pose, PoseComposerService, PoseExtractorService };
//# sourceMappingURL=ngx-mp-pose-extractor.mjs.map
