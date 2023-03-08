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

class ImageTrimmer {
    constructor() { }
    loadByDataUrl(dataUrl) {
        return __awaiter(this, void 0, void 0, function* () {
            const image = yield new Promise((resolve, reject) => {
                const image = new Image();
                image.src = dataUrl;
                image.onload = () => {
                    resolve(image);
                };
            });
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = image.width;
            canvas.height = image.height;
            context.drawImage(image, 0, 0);
            this.canvas = canvas;
            this.context = context;
        });
    }
    trimMargin(marginColor, diffThreshold = 10) {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.canvas === undefined)
                throw new Error('Image is not loaded');
            // マージンを検出する範囲を指定 (左端から0〜20%)
            const edgeDetectionRangeMinX = 0;
            const edgeDetectionRangeMaxX = Math.floor(this.canvas.width * 0.2);
            // マージンの端を検出
            const edgePositionFromTop = yield this.getVerticalEdgePositionOfColor(marginColor, 'top', diffThreshold, edgeDetectionRangeMinX, edgeDetectionRangeMaxX);
            const marginTop = edgePositionFromTop != null ? edgePositionFromTop : 0;
            const edgePositionFromBottom = yield this.getVerticalEdgePositionOfColor(marginColor, 'bottom', diffThreshold, edgeDetectionRangeMinX, edgeDetectionRangeMaxX);
            const marginBottom = edgePositionFromBottom != null
                ? edgePositionFromBottom
                : this.canvas.height;
            const oldHeight = this.canvas.height;
            const newHeight = marginBottom - marginTop;
            this.crop(0, marginTop, this.canvas.width, newHeight);
            return {
                marginTop: marginTop,
                marginBottom: marginBottom,
                heightNew: newHeight,
                heightOld: oldHeight,
                width: this.canvas.width,
            };
        });
    }
    crop(x, y, w, h) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.canvas || !this.context)
                return;
            const newCanvas = document.createElement('canvas');
            const newContext = newCanvas.getContext('2d');
            newCanvas.width = w;
            newCanvas.height = h;
            newContext.drawImage(this.canvas, x, y, newCanvas.width, newCanvas.height, 0, 0, newCanvas.width, newCanvas.height);
            this.replaceCanvas(newCanvas);
        });
    }
    replaceColor(srcColor, dstColor, diffThreshold) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.canvas || !this.context)
                return;
            const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
            const dstColorValue = this.hexColorCodeToRgba(dstColor);
            for (let x = 0; x < imageData.width; x++) {
                for (let y = 0; y < imageData.height; y++) {
                    const idx = (x + y * imageData.width) * 4;
                    const red = imageData.data[idx + 0];
                    const green = imageData.data[idx + 1];
                    const blue = imageData.data[idx + 2];
                    const alpha = imageData.data[idx + 3];
                    const colorCode = this.rgbToHexColorCode(red, green, blue);
                    if (!this.isSimilarColor(srcColor, colorCode, diffThreshold)) {
                        continue;
                    }
                    imageData.data[idx + 0] = dstColorValue.r;
                    imageData.data[idx + 1] = dstColorValue.g;
                    imageData.data[idx + 2] = dstColorValue.b;
                    if (dstColorValue.a !== undefined) {
                        imageData.data[idx + 3] = dstColorValue.a;
                    }
                }
            }
            this.context.putImageData(imageData, 0, 0);
        });
    }
    getMarginColor() {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.canvas || !this.context) {
                return null;
            }
            let marginColor = null;
            const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
            let isBreak = false;
            for (let x = 0; x < imageData.width && !isBreak; x++) {
                for (let y = 0; y < imageData.height; y++) {
                    const idx = (x + y * imageData.width) * 4;
                    const red = imageData.data[idx + 0];
                    const green = imageData.data[idx + 1];
                    const blue = imageData.data[idx + 2];
                    const alpha = imageData.data[idx + 3];
                    const colorCode = this.rgbToHexColorCode(red, green, blue);
                    if (marginColor != colorCode) {
                        if (marginColor === null) {
                            marginColor = colorCode;
                        }
                        else {
                            isBreak = true;
                            break;
                        }
                    }
                }
            }
            return marginColor;
        });
    }
    getVerticalEdgePositionOfColor(color, direction, diffThreshold, minX, maxX) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.canvas || !this.context) {
                return null;
            }
            if (minX === undefined) {
                minX = 0;
            }
            if (maxX === undefined) {
                maxX = this.canvas.width;
            }
            const imageData = this.context.getImageData(0, 0, this.canvas.width, this.canvas.height);
            let isBreak = false;
            let edgePositionY;
            if (direction === 'top') {
                edgePositionY = 0;
                for (let y = 0; y < imageData.height; y++) {
                    for (let x = 0; x < maxX && !isBreak; x++) {
                        const idx = (x + y * imageData.width) * 4;
                        const red = imageData.data[idx + 0];
                        const green = imageData.data[idx + 1];
                        const blue = imageData.data[idx + 2];
                        const alpha = imageData.data[idx + 3];
                        const colorCode = this.rgbToHexColorCode(red, green, blue);
                        if (color == colorCode ||
                            this.isSimilarColor(color, colorCode, diffThreshold)) {
                            if (edgePositionY < y) {
                                edgePositionY = y;
                            }
                        }
                        else {
                            isBreak = true;
                            break;
                        }
                    }
                }
                if (edgePositionY !== 0) {
                    edgePositionY += 1;
                }
            }
            else if (direction === 'bottom') {
                edgePositionY = this.canvas.height;
                for (let y = imageData.height - 1; y >= 0; y--) {
                    for (let x = 0; x < imageData.width && !isBreak; x++) {
                        const idx = (x + y * imageData.width) * 4;
                        const red = imageData.data[idx + 0];
                        const green = imageData.data[idx + 1];
                        const blue = imageData.data[idx + 2];
                        const alpha = imageData.data[idx + 3];
                        const colorCode = this.rgbToHexColorCode(red, green, blue);
                        if (color == colorCode ||
                            this.isSimilarColor(color, colorCode, diffThreshold)) {
                            if (edgePositionY > y) {
                                edgePositionY = y;
                            }
                        }
                        else {
                            isBreak = true;
                            break;
                        }
                    }
                }
                if (edgePositionY !== this.canvas.height) {
                    edgePositionY -= 1;
                }
            }
            return edgePositionY;
        });
    }
    getWidth() {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            return (_a = this.canvas) === null || _a === void 0 ? void 0 : _a.width;
        });
    }
    getHeight() {
        var _a;
        return __awaiter(this, void 0, void 0, function* () {
            return (_a = this.canvas) === null || _a === void 0 ? void 0 : _a.height;
        });
    }
    resizeWithFit(param) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.canvas) {
                return;
            }
            let newWidth = 0, newHeight = 0;
            if (param.width && this.canvas.width > param.width) {
                newWidth = param.width ? param.width : this.canvas.width;
                newHeight = this.canvas.height * (newWidth / this.canvas.width);
            }
            else if (param.height && this.canvas.height > param.height) {
                newHeight = param.height ? param.height : this.canvas.height;
                newWidth = this.canvas.width * (newHeight / this.canvas.height);
            }
            else {
                return;
            }
            const newCanvas = document.createElement('canvas');
            const newContext = newCanvas.getContext('2d');
            newCanvas.width = newWidth;
            newCanvas.height = newHeight;
            newContext.drawImage(this.canvas, 0, 0, newWidth, newHeight);
            this.replaceCanvas(newCanvas);
        });
    }
    replaceCanvas(canvas) {
        if (!this.canvas || !this.context) {
            this.canvas = canvas;
            this.context = this.canvas.getContext('2d');
            return;
        }
        this.canvas.width = 0;
        this.canvas.height = 0;
        delete this.canvas;
        delete this.context;
        this.canvas = canvas;
        this.context = this.canvas.getContext('2d');
    }
    getDataUrl(mime = 'image/jpeg', imageQuality) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.canvas) {
                return null;
            }
            if (mime === 'image/jpeg' || mime === 'image/webp') {
                return this.canvas.toDataURL(mime, imageQuality);
            }
            else {
                return this.canvas.toDataURL(mime);
            }
        });
    }
    hexColorCodeToRgba(color) {
        const r = parseInt(color.substr(1, 2), 16);
        const g = parseInt(color.substr(3, 2), 16);
        const b = parseInt(color.substr(5, 2), 16);
        if (color.length === 9) {
            const a = parseInt(color.substr(7, 2), 16);
            return { r, g, b, a };
        }
        return { r, g, b };
    }
    rgbToHexColorCode(r, g, b) {
        return '#' + this.valueToHex(r) + this.valueToHex(g) + this.valueToHex(b);
    }
    valueToHex(value) {
        return ('0' + value.toString(16)).slice(-2);
    }
    isSimilarColor(color1, color2, diffThreshold) {
        const color1Rgb = this.hexColorCodeToRgba(color1);
        const color2Rgb = this.hexColorCodeToRgba(color2);
        const diff = Math.abs(color1Rgb.r - color2Rgb.r) +
            Math.abs(color1Rgb.g - color2Rgb.g) +
            Math.abs(color1Rgb.b - color2Rgb.b);
        return diff < diffThreshold;
    }
}

class PoseSet {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        // ポーズを追加するためのキュー
        this.similarPoseQueue = [];
        // 類似ポーズの除去 - 全ポーズから
        this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_WHOLE = false;
        // 類似ポーズの除去 - 各ポーズの前後から
        this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND = true;
        // 画像書き出し時の設定
        this.IMAGE_WIDTH = 1080;
        this.IMAGE_MIME = 'image/webp';
        this.IMAGE_QUALITY = 0.8;
        // 画像の余白除去
        this.IMAGE_MARGIN_TRIMMING_COLOR = '#000000';
        this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD = 50;
        // 画像の背景色置換
        this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR = '#016AFD';
        this.IMAGE_BACKGROUND_REPLACE_DST_COLOR = '#FFFFFF00';
        this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD = 130;
        this.videoMetadata = {
            name: '',
            width: 0,
            height: 0,
            duration: 0,
            firstPoseDetectedTime: 0,
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
    /**
     * ポーズ数の取得
     * @returns
     */
    getNumberOfPoses() {
        if (this.poses === undefined)
            return -1;
        return this.poses.length;
    }
    /**
     * 全ポーズの取得
     * @returns 全てのポーズ
     */
    getPoses() {
        if (this.poses === undefined)
            return [];
        return this.poses;
    }
    /**
     * 指定された時間によるポーズの取得
     * @param timeMiliseconds ポーズの時間 (ミリ秒)
     * @returns ポーズ
     */
    getPoseByTime(timeMiliseconds) {
        if (this.poses === undefined)
            return undefined;
        return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
    }
    /**
     * ポーズの追加
     */
    pushPose(videoTimeMiliseconds, frameImageDataUrl, poseImageDataUrl, faceFrameImageDataUrl, results) {
        var _a, _b;
        if (results.poseLandmarks === undefined)
            return;
        if (this.poses.length === 0) {
            this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
        }
        const poseLandmarksWithWorldCoordinate = results.ea
            ? results.ea
            : [];
        if (poseLandmarksWithWorldCoordinate.length === 0) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the pose with the world coordinate`, results);
            return;
        }
        const bodyVector = PoseSet.getBodyVector(poseLandmarksWithWorldCoordinate);
        if (!bodyVector) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the body vector`, poseLandmarksWithWorldCoordinate);
            return;
        }
        if (results.leftHandLandmarks === undefined &&
            results.rightHandLandmarks === undefined) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand landmarks`, results);
        }
        else if (results.leftHandLandmarks === undefined) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the left hand landmarks`, results);
        }
        else if (results.rightHandLandmarks === undefined) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the right hand landmarks`, results);
        }
        const handVector = PoseSet.getHandVector(results.leftHandLandmarks, results.rightHandLandmarks);
        if (!handVector) {
            console.warn(`[PoseSet] pushPose (${videoTimeMiliseconds}) - Could not get the hand vector`, results);
        }
        const pose = {
            timeMiliseconds: videoTimeMiliseconds,
            durationMiliseconds: -1,
            pose: poseLandmarksWithWorldCoordinate.map((worldCoordinateLandmark) => {
                return [
                    worldCoordinateLandmark.x,
                    worldCoordinateLandmark.y,
                    worldCoordinateLandmark.z,
                    worldCoordinateLandmark.visibility,
                ];
            }),
            leftHand: (_a = results.leftHandLandmarks) === null || _a === void 0 ? void 0 : _a.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            rightHand: (_b = results.rightHandLandmarks) === null || _b === void 0 ? void 0 : _b.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            bodyVector: bodyVector,
            handVector: handVector,
            frameImageDataUrl: frameImageDataUrl,
            poseImageDataUrl: poseImageDataUrl,
            faceFrameImageDataUrl: faceFrameImageDataUrl,
            extendedData: {},
            debug: {
                duplicatedItems: [],
            },
            mergedTimeMiliseconds: -1,
            mergedDurationMiliseconds: -1,
        };
        let lastPose;
        if (this.poses.length === 0 && 1 <= this.similarPoseQueue.length) {
            // 類似ポーズキューから最後のポーズを取得
            lastPose = this.similarPoseQueue[this.similarPoseQueue.length - 1];
        }
        else if (1 <= this.poses.length) {
            // ポーズ配列から最後のポーズを取得
            lastPose = this.poses[this.poses.length - 1];
        }
        if (lastPose) {
            // 最後のポーズがあれば、類似ポーズかどうかを比較
            const isSimilarBodyPose = PoseSet.isSimilarBodyPose(pose.bodyVector, lastPose.bodyVector);
            let isSimilarHandPose = true;
            if (lastPose.handVector && pose.handVector) {
                isSimilarHandPose = PoseSet.isSimilarHandPose(pose.handVector, lastPose.handVector);
            }
            else if (!lastPose.handVector && pose.handVector) {
                isSimilarHandPose = false;
            }
            if (!isSimilarBodyPose || !isSimilarHandPose) {
                // 身体・手のいずれかが前のポーズと類似していないならば、類似ポーズキューを処理して、ポーズ配列へ追加
                this.pushPoseFromSimilarPoseQueue(pose.timeMiliseconds);
            }
        }
        // 類似ポーズキューへ追加
        this.similarPoseQueue.push(pose);
        return pose;
    }
    /**
     * ポーズの配列からポーズが決まっている瞬間を取得
     * @param poses ポーズの配列
     * @returns ポーズが決まっている瞬間
     */
    static getSuitablePoseByPoses(poses) {
        if (poses.length === 0)
            return null;
        if (poses.length === 1) {
            return poses[1];
        }
        // 各標本ポーズごとの類似度を初期化
        const similaritiesOfPoses = {};
        for (let i = 0; i < poses.length; i++) {
            similaritiesOfPoses[poses[i].timeMiliseconds] = poses.map((pose) => {
                return {
                    handSimilarity: 0,
                    bodySimilarity: 0,
                };
            });
        }
        // 各標本ポーズごとの類似度を計算
        for (let samplePose of poses) {
            let handSimilarity;
            for (let i = 0; i < poses.length; i++) {
                const pose = poses[i];
                if (pose.handVector && samplePose.handVector) {
                    handSimilarity = PoseSet.getHandSimilarity(pose.handVector, samplePose.handVector);
                }
                let bodySimilarity = PoseSet.getBodyPoseSimilarity(pose.bodyVector, samplePose.bodyVector);
                similaritiesOfPoses[samplePose.timeMiliseconds][i] = {
                    handSimilarity: handSimilarity !== null && handSimilarity !== void 0 ? handSimilarity : 0,
                    bodySimilarity,
                };
            }
        }
        // 類似度の高いフレームが多かったポーズを選択
        const similaritiesOfSamplePoses = poses.map((pose) => {
            return similaritiesOfPoses[pose.timeMiliseconds].reduce((prev, current) => {
                return prev + current.handSimilarity + current.bodySimilarity;
            }, 0);
        });
        const maxSimilarity = Math.max(...similaritiesOfSamplePoses);
        const maxSimilarityIndex = similaritiesOfSamplePoses.indexOf(maxSimilarity);
        const selectedPose = poses[maxSimilarityIndex];
        if (!selectedPose) {
            console.warn(`[PoseSet] getSuitablePoseByPoses`, similaritiesOfSamplePoses, maxSimilarity, maxSimilarityIndex);
        }
        console.debug(`[PoseSet] getSuitablePoseByPoses`, {
            selected: selectedPose,
            unselected: poses.filter((pose) => {
                return pose.timeMiliseconds !== selectedPose.timeMiliseconds;
            }),
        });
        return selectedPose;
    }
    /**
     * 最終処理
     * (重複したポーズの除去、画像のマージン除去など)
     */
    finalize() {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.similarPoseQueue.length > 0) {
                // 類似ポーズキューにポーズが残っている場合、最適なポーズを選択してポーズ配列へ追加
                this.pushPoseFromSimilarPoseQueue(this.videoMetadata.duration);
            }
            if (0 == this.poses.length) {
                // ポーズが一つもない場合、処理を終了
                this.isFinalized = true;
                return;
            }
            // ポーズの持続時間を設定
            for (let i = 0; i < this.poses.length - 1; i++) {
                if (this.poses[i].durationMiliseconds !== -1)
                    continue;
                this.poses[i].durationMiliseconds =
                    this.poses[i + 1].timeMiliseconds - this.poses[i].timeMiliseconds;
            }
            this.poses[this.poses.length - 1].durationMiliseconds =
                this.videoMetadata.duration -
                    this.poses[this.poses.length - 1].timeMiliseconds;
            // 全体から重複ポーズを除去
            if (this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_WHOLE) {
                this.removeDuplicatedPoses();
            }
            // 最初のポーズを除去
            this.poses.shift();
            // 画像のマージンを取得
            console.debug(`[PoseSet] finalize - Detecting image margins...`);
            let imageTrimming = undefined;
            for (const pose of this.poses) {
                let imageTrimmer = new ImageTrimmer();
                if (!pose.frameImageDataUrl) {
                    continue;
                }
                yield imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
                const marginColor = yield imageTrimmer.getMarginColor();
                console.debug(`[PoseSet] finalize - Detected margin color...`, pose.timeMiliseconds, marginColor);
                if (marginColor === null)
                    continue;
                if (marginColor !== this.IMAGE_MARGIN_TRIMMING_COLOR) {
                    continue;
                }
                const trimmed = yield imageTrimmer.trimMargin(marginColor, this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD);
                if (!trimmed)
                    continue;
                imageTrimming = trimmed;
                console.debug(`[PoseSet] finalize - Determined image trimming positions...`, trimmed);
                break;
            }
            // 画像を整形
            for (const pose of this.poses) {
                let imageTrimmer = new ImageTrimmer();
                if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
                    continue;
                }
                console.debug(`[PoseSet] finalize - Processing image...`, pose.timeMiliseconds);
                // 画像を整形 - フレーム画像
                yield imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
                if (imageTrimming) {
                    yield imageTrimmer.crop(0, imageTrimming.marginTop, imageTrimming.width, imageTrimming.heightNew);
                }
                yield imageTrimmer.replaceColor(this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR, this.IMAGE_BACKGROUND_REPLACE_DST_COLOR, this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD);
                yield imageTrimmer.resizeWithFit({
                    width: this.IMAGE_WIDTH,
                });
                let newDataUrl = yield imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                    ? this.IMAGE_QUALITY
                    : undefined);
                if (!newDataUrl) {
                    console.warn(`[PoseSet] finalize - Could not get the new dataurl for frame image`);
                    continue;
                }
                pose.frameImageDataUrl = newDataUrl;
                // 画像を整形 - ポーズプレビュー画像
                imageTrimmer = new ImageTrimmer();
                yield imageTrimmer.loadByDataUrl(pose.poseImageDataUrl);
                if (imageTrimming) {
                    yield imageTrimmer.crop(0, imageTrimming.marginTop, imageTrimming.width, imageTrimming.heightNew);
                }
                yield imageTrimmer.resizeWithFit({
                    width: this.IMAGE_WIDTH,
                });
                newDataUrl = yield imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                    ? this.IMAGE_QUALITY
                    : undefined);
                if (!newDataUrl) {
                    console.warn(`[PoseSet] finalize - Could not get the new dataurl for pose preview image`);
                    continue;
                }
                pose.poseImageDataUrl = newDataUrl;
                if (pose.faceFrameImageDataUrl) {
                    // 画像を整形 - 顔フレーム画像
                    imageTrimmer = new ImageTrimmer();
                    yield imageTrimmer.loadByDataUrl(pose.faceFrameImageDataUrl);
                    newDataUrl = yield imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                        ? this.IMAGE_QUALITY
                        : undefined);
                    if (!newDataUrl) {
                        console.warn(`[PoseSet] finalize - Could not get the new dataurl for face frame image`);
                        continue;
                    }
                    pose.faceFrameImageDataUrl = newDataUrl;
                }
            }
            this.isFinalized = true;
        });
    }
    /**
     * 類似ポーズの取得
     * @param results MediaPipe Holistic によるポーズの検出結果
     * @param threshold しきい値
     * @param targetRange ポーズを比較する範囲 (all: 全て, bodyPose: 身体のみ, handPose: 手指のみ)
     * @returns 類似ポーズの配列
     */
    getSimilarPoses(results, threshold = 0.9, targetRange = 'all') {
        // 身体のベクトルを取得
        let bodyVector;
        try {
            bodyVector = PoseSet.getBodyVector(results.ea);
        }
        catch (e) {
            console.error(`[PoseSet] getSimilarPoses - Error occurred`, e, results);
            return [];
        }
        if (!bodyVector) {
            throw 'Could not get the body vector';
        }
        // 手指のベクトルを取得
        let handVector;
        if (targetRange === 'all' || targetRange === 'handPose') {
            handVector = PoseSet.getHandVector(results.leftHandLandmarks, results.rightHandLandmarks);
            if (targetRange === 'handPose' && !handVector) {
                throw 'Could not get the hand vector';
            }
        }
        // 各ポーズとベクトルを比較
        const poses = [];
        for (const pose of this.poses) {
            if ((targetRange === 'all' || targetRange === 'bodyPose') &&
                !pose.bodyVector) {
                continue;
            }
            else if (targetRange === 'handPose' && !pose.handVector) {
                continue;
            }
            /*console.debug(
              '[PoseSet] getSimilarPoses - ',
              this.getVideoName(),
              pose.timeMiliseconds
            );*/
            // 身体のポーズの類似度を取得
            let bodySimilarity;
            if (bodyVector && pose.bodyVector) {
                bodySimilarity = PoseSet.getBodyPoseSimilarity(pose.bodyVector, bodyVector);
            }
            // 手指のポーズの類似度を取得
            let handSimilarity;
            if (handVector && pose.handVector) {
                handSimilarity = PoseSet.getHandSimilarity(pose.handVector, handVector);
            }
            // 判定
            let similarity, isSimilar = false;
            if (targetRange === 'all') {
                similarity = Math.max(bodySimilarity !== null && bodySimilarity !== void 0 ? bodySimilarity : 0, handSimilarity !== null && handSimilarity !== void 0 ? handSimilarity : 0);
                if (threshold <= bodySimilarity || threshold <= handSimilarity) {
                    isSimilar = true;
                }
            }
            else if (targetRange === 'bodyPose') {
                similarity = bodySimilarity;
                if (threshold <= bodySimilarity) {
                    isSimilar = true;
                }
            }
            else if (targetRange === 'handPose') {
                similarity = handSimilarity;
                if (threshold <= handSimilarity) {
                    isSimilar = true;
                }
            }
            if (!isSimilar)
                continue;
            // 結果へ追加
            poses.push(Object.assign(Object.assign({}, pose), { similarity: similarity, bodyPoseSimilarity: bodySimilarity, handPoseSimilarity: handSimilarity }));
        }
        return poses;
    }
    /**
     * 身体の姿勢を表すベクトルの取得
     * @param poseLandmarks MediaPipe Holistic で取得できた身体のワールド座標 (ra 配列)
     * @returns ベクトル
     */
    static getBodyVector(poseLandmarks) {
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
    /**
     * 手指の姿勢を表すベクトルの取得
     * @param leftHandLandmarks MediaPipe Holistic で取得できた左手の正規化座標
     * @param rightHandLandmarks MediaPipe Holistic で取得できた右手の正規化座標
     * @returns ベクトル
     */
    static getHandVector(leftHandLandmarks, rightHandLandmarks) {
        if ((rightHandLandmarks === undefined || rightHandLandmarks.length === 0) &&
            (leftHandLandmarks === undefined || leftHandLandmarks.length === 0)) {
            return undefined;
        }
        return {
            // 右手 - 親指
            rightThumbTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[4].x - rightHandLandmarks[3].x,
                    rightHandLandmarks[4].y - rightHandLandmarks[3].y,
                    rightHandLandmarks[4].z - rightHandLandmarks[3].z,
                ],
            rightThumbFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[3].x - rightHandLandmarks[2].x,
                    rightHandLandmarks[3].y - rightHandLandmarks[2].y,
                    rightHandLandmarks[3].z - rightHandLandmarks[2].z,
                ],
            // 右手 - 人差し指
            rightIndexFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[8].x - rightHandLandmarks[7].x,
                    rightHandLandmarks[8].y - rightHandLandmarks[7].y,
                    rightHandLandmarks[8].z - rightHandLandmarks[7].z,
                ],
            rightIndexFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[7].x - rightHandLandmarks[6].x,
                    rightHandLandmarks[7].y - rightHandLandmarks[6].y,
                    rightHandLandmarks[7].z - rightHandLandmarks[6].z,
                ],
            // 右手 - 中指
            rightMiddleFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[12].x - rightHandLandmarks[11].x,
                    rightHandLandmarks[12].y - rightHandLandmarks[11].y,
                    rightHandLandmarks[12].z - rightHandLandmarks[11].z,
                ],
            rightMiddleFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[11].x - rightHandLandmarks[10].x,
                    rightHandLandmarks[11].y - rightHandLandmarks[10].y,
                    rightHandLandmarks[11].z - rightHandLandmarks[10].z,
                ],
            // 右手 - 薬指
            rightRingFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[16].x - rightHandLandmarks[15].x,
                    rightHandLandmarks[16].y - rightHandLandmarks[15].y,
                    rightHandLandmarks[16].z - rightHandLandmarks[15].z,
                ],
            rightRingFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[15].x - rightHandLandmarks[14].x,
                    rightHandLandmarks[15].y - rightHandLandmarks[14].y,
                    rightHandLandmarks[15].z - rightHandLandmarks[14].z,
                ],
            // 右手 - 小指
            rightPinkyFingerTipToFirstJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[20].x - rightHandLandmarks[19].x,
                    rightHandLandmarks[20].y - rightHandLandmarks[19].y,
                    rightHandLandmarks[20].z - rightHandLandmarks[19].z,
                ],
            rightPinkyFingerFirstJointToSecondJoint: rightHandLandmarks === undefined || rightHandLandmarks.length === 0
                ? null
                : [
                    rightHandLandmarks[19].x - rightHandLandmarks[18].x,
                    rightHandLandmarks[19].y - rightHandLandmarks[18].y,
                    rightHandLandmarks[19].z - rightHandLandmarks[18].z,
                ],
            // 左手 - 親指
            leftThumbTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[4].x - leftHandLandmarks[3].x,
                    leftHandLandmarks[4].y - leftHandLandmarks[3].y,
                    leftHandLandmarks[4].z - leftHandLandmarks[3].z,
                ],
            leftThumbFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[3].x - leftHandLandmarks[2].x,
                    leftHandLandmarks[3].y - leftHandLandmarks[2].y,
                    leftHandLandmarks[3].z - leftHandLandmarks[2].z,
                ],
            // 左手 - 人差し指
            leftIndexFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[8].x - leftHandLandmarks[7].x,
                    leftHandLandmarks[8].y - leftHandLandmarks[7].y,
                    leftHandLandmarks[8].z - leftHandLandmarks[7].z,
                ],
            leftIndexFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[7].x - leftHandLandmarks[6].x,
                    leftHandLandmarks[7].y - leftHandLandmarks[6].y,
                    leftHandLandmarks[7].z - leftHandLandmarks[6].z,
                ],
            // 左手 - 中指
            leftMiddleFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[12].x - leftHandLandmarks[11].x,
                    leftHandLandmarks[12].y - leftHandLandmarks[11].y,
                    leftHandLandmarks[12].z - leftHandLandmarks[11].z,
                ],
            leftMiddleFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[11].x - leftHandLandmarks[10].x,
                    leftHandLandmarks[11].y - leftHandLandmarks[10].y,
                    leftHandLandmarks[11].z - leftHandLandmarks[10].z,
                ],
            // 左手 - 薬指
            leftRingFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[16].x - leftHandLandmarks[15].x,
                    leftHandLandmarks[16].y - leftHandLandmarks[15].y,
                    leftHandLandmarks[16].z - leftHandLandmarks[15].z,
                ],
            leftRingFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[15].x - leftHandLandmarks[14].x,
                    leftHandLandmarks[15].y - leftHandLandmarks[14].y,
                    leftHandLandmarks[15].z - leftHandLandmarks[14].z,
                ],
            // 左手 - 小指
            leftPinkyFingerTipToFirstJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[20].x - leftHandLandmarks[19].x,
                    leftHandLandmarks[20].y - leftHandLandmarks[19].y,
                    leftHandLandmarks[20].z - leftHandLandmarks[19].z,
                ],
            leftPinkyFingerFirstJointToSecondJoint: leftHandLandmarks === undefined || leftHandLandmarks.length === 0
                ? null
                : [
                    leftHandLandmarks[19].x - leftHandLandmarks[18].x,
                    leftHandLandmarks[19].y - leftHandLandmarks[18].y,
                    leftHandLandmarks[19].z - leftHandLandmarks[18].z,
                ],
        };
    }
    /**
     * BodyVector 間が類似しているかどうかの判定
     * @param bodyVectorA 比較先の BodyVector
     * @param bodyVectorB 比較元の BodyVector
     * @param threshold しきい値
     * @returns 類似しているかどうか
     */
    static isSimilarBodyPose(bodyVectorA, bodyVectorB, threshold = 0.8) {
        let isSimilar = false;
        const similarity = PoseSet.getBodyPoseSimilarity(bodyVectorA, bodyVectorB);
        if (similarity >= threshold)
            isSimilar = true;
        // console.debug(`[PoseSet] isSimilarPose`, isSimilar, similarity);
        return isSimilar;
    }
    /**
     * 身体ポーズの類似度の取得
     * @param bodyVectorA 比較先の BodyVector
     * @param bodyVectorB 比較元の BodyVector
     * @returns 類似度
     */
    static getBodyPoseSimilarity(bodyVectorA, bodyVectorB) {
        const cosSimilarities = {
            leftWristToLeftElbow: cosSimilarity(bodyVectorA.leftWristToLeftElbow, bodyVectorB.leftWristToLeftElbow),
            leftElbowToLeftShoulder: cosSimilarity(bodyVectorA.leftElbowToLeftShoulder, bodyVectorB.leftElbowToLeftShoulder),
            rightWristToRightElbow: cosSimilarity(bodyVectorA.rightWristToRightElbow, bodyVectorB.rightWristToRightElbow),
            rightElbowToRightShoulder: cosSimilarity(bodyVectorA.rightElbowToRightShoulder, bodyVectorB.rightElbowToRightShoulder),
        };
        const cosSimilaritiesSum = Object.values(cosSimilarities).reduce((sum, value) => sum + value, 0);
        return cosSimilaritiesSum / Object.keys(cosSimilarities).length;
    }
    /**
     * HandVector 間が類似しているかどうかの判定
     * @param handVectorA 比較先の HandVector
     * @param handVectorB 比較元の HandVector
     * @param threshold しきい値
     * @returns 類似しているかどうか
     */
    static isSimilarHandPose(handVectorA, handVectorB, threshold = 0.75) {
        const similarity = PoseSet.getHandSimilarity(handVectorA, handVectorB);
        if (similarity === -1) {
            return true;
        }
        return similarity >= threshold;
    }
    /**
     * 手のポーズの類似度の取得
     * @param handVectorA 比較先の HandVector
     * @param handVectorB 比較元の HandVector
     * @returns 類似度
     */
    static getHandSimilarity(handVectorA, handVectorB) {
        const cosSimilaritiesRightHand = handVectorA.rightThumbFirstJointToSecondJoint === null ||
            handVectorB.rightThumbFirstJointToSecondJoint === null
            ? undefined
            : {
                // 右手 - 親指
                rightThumbTipToFirstJoint: cosSimilarity(handVectorA.rightThumbTipToFirstJoint, handVectorB.rightThumbTipToFirstJoint),
                rightThumbFirstJointToSecondJoint: cosSimilarity(handVectorA.rightThumbFirstJointToSecondJoint, handVectorB.rightThumbFirstJointToSecondJoint),
                // 右手 - 人差し指
                rightIndexFingerTipToFirstJoint: cosSimilarity(handVectorA.rightIndexFingerTipToFirstJoint, handVectorB.rightIndexFingerTipToFirstJoint),
                rightIndexFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightIndexFingerFirstJointToSecondJoint, handVectorB.rightIndexFingerFirstJointToSecondJoint),
                // 右手 - 中指
                rightMiddleFingerTipToFirstJoint: cosSimilarity(handVectorA.rightMiddleFingerTipToFirstJoint, handVectorB.rightMiddleFingerTipToFirstJoint),
                rightMiddleFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightMiddleFingerFirstJointToSecondJoint, handVectorB.rightMiddleFingerFirstJointToSecondJoint),
                // 右手 - 薬指
                rightRingFingerTipToFirstJoint: cosSimilarity(handVectorA.rightRingFingerTipToFirstJoint, handVectorB.rightRingFingerFirstJointToSecondJoint),
                rightRingFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightRingFingerFirstJointToSecondJoint, handVectorB.rightRingFingerFirstJointToSecondJoint),
                // 右手 - 小指
                rightPinkyFingerTipToFirstJoint: cosSimilarity(handVectorA.rightPinkyFingerTipToFirstJoint, handVectorB.rightPinkyFingerTipToFirstJoint),
                rightPinkyFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.rightPinkyFingerFirstJointToSecondJoint, handVectorB.rightPinkyFingerFirstJointToSecondJoint),
            };
        const cosSimilaritiesLeftHand = handVectorA.leftThumbFirstJointToSecondJoint === null ||
            handVectorB.leftThumbFirstJointToSecondJoint === null
            ? undefined
            : {
                // 左手 - 親指
                leftThumbTipToFirstJoint: cosSimilarity(handVectorA.leftThumbTipToFirstJoint, handVectorB.leftThumbTipToFirstJoint),
                leftThumbFirstJointToSecondJoint: cosSimilarity(handVectorA.leftThumbFirstJointToSecondJoint, handVectorB.leftThumbFirstJointToSecondJoint),
                // 左手 - 人差し指
                leftIndexFingerTipToFirstJoint: cosSimilarity(handVectorA.leftIndexFingerTipToFirstJoint, handVectorB.leftIndexFingerTipToFirstJoint),
                leftIndexFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftIndexFingerFirstJointToSecondJoint, handVectorB.leftIndexFingerFirstJointToSecondJoint),
                // 左手 - 中指
                leftMiddleFingerTipToFirstJoint: cosSimilarity(handVectorA.leftMiddleFingerTipToFirstJoint, handVectorB.leftMiddleFingerTipToFirstJoint),
                leftMiddleFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftMiddleFingerFirstJointToSecondJoint, handVectorB.leftMiddleFingerFirstJointToSecondJoint),
                // 左手 - 薬指
                leftRingFingerTipToFirstJoint: cosSimilarity(handVectorA.leftRingFingerTipToFirstJoint, handVectorB.leftRingFingerTipToFirstJoint),
                leftRingFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftRingFingerFirstJointToSecondJoint, handVectorB.leftRingFingerFirstJointToSecondJoint),
                // 左手 - 小指
                leftPinkyFingerTipToFirstJoint: cosSimilarity(handVectorA.leftPinkyFingerTipToFirstJoint, handVectorB.leftPinkyFingerTipToFirstJoint),
                leftPinkyFingerFirstJointToSecondJoint: cosSimilarity(handVectorA.leftPinkyFingerFirstJointToSecondJoint, handVectorB.leftPinkyFingerFirstJointToSecondJoint),
            };
        // 左手の類似度
        let cosSimilaritiesSumLeftHand = 0;
        if (cosSimilaritiesLeftHand) {
            cosSimilaritiesSumLeftHand = Object.values(cosSimilaritiesLeftHand).reduce((sum, value) => sum + value, 0);
        }
        // 右手の類似度
        let cosSimilaritiesSumRightHand = 0;
        if (cosSimilaritiesRightHand) {
            cosSimilaritiesSumRightHand = Object.values(cosSimilaritiesRightHand).reduce((sum, value) => sum + value, 0);
        }
        // 合算された類似度
        if (cosSimilaritiesRightHand && cosSimilaritiesLeftHand) {
            return ((cosSimilaritiesSumRightHand + cosSimilaritiesSumLeftHand) /
                (Object.keys(cosSimilaritiesRightHand).length +
                    Object.keys(cosSimilaritiesLeftHand).length));
        }
        else if (cosSimilaritiesRightHand) {
            if (handVectorB.leftThumbFirstJointToSecondJoint !== null &&
                handVectorA.leftThumbFirstJointToSecondJoint === null) {
                // handVectorB で左手があるのに handVectorA で左手がない場合、類似度を減らす
                console.debug(`[PoseSet] getHandSimilarity - Adjust similarity, because left hand not found...`);
                return (cosSimilaritiesSumRightHand /
                    (Object.keys(cosSimilaritiesRightHand).length * 2));
            }
            return (cosSimilaritiesSumRightHand /
                Object.keys(cosSimilaritiesRightHand).length);
        }
        else if (cosSimilaritiesLeftHand) {
            if (handVectorB.rightThumbFirstJointToSecondJoint !== null &&
                handVectorA.rightThumbFirstJointToSecondJoint === null) {
                // handVectorB で右手があるのに handVectorA で右手がない場合、類似度を減らす
                console.debug(`[PoseSet] getHandSimilarity - Adjust similarity, because right hand not found...`);
                return (cosSimilaritiesSumLeftHand /
                    (Object.keys(cosSimilaritiesLeftHand).length * 2));
            }
            return (cosSimilaritiesSumLeftHand /
                Object.keys(cosSimilaritiesLeftHand).length);
        }
        return -1;
    }
    /**
     * ZIP ファイルとしてのシリアライズ
     * @returns ZIPファイル (Blob 形式)
     */
    getZip() {
        return __awaiter(this, void 0, void 0, function* () {
            const jsZip = new JSZip();
            jsZip.file('poses.json', yield this.getJson());
            const imageFileExt = this.getFileExtensionByMime(this.IMAGE_MIME);
            for (const pose of this.poses) {
                if (pose.frameImageDataUrl) {
                    try {
                        const index = pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                        const base64 = pose.frameImageDataUrl.substring(index);
                        jsZip.file(`frame-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
                            base64: true,
                        });
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
                        jsZip.file(`pose-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
                            base64: true,
                        });
                    }
                    catch (error) {
                        console.warn(`[PoseExporterService] push - Could not push frame image`, error);
                        throw error;
                    }
                }
                if (pose.faceFrameImageDataUrl) {
                    try {
                        const index = pose.faceFrameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                        const base64 = pose.faceFrameImageDataUrl.substring(index);
                        jsZip.file(`face-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
                            base64: true,
                        });
                    }
                    catch (error) {
                        console.warn(`[PoseExporterService] push - Could not push face frame image`, error);
                        throw error;
                    }
                }
            }
            return yield jsZip.generateAsync({ type: 'blob' });
        });
    }
    /**
     * JSON 文字列としてのシリアライズ
     * @returns JSON 文字列
     */
    getJson() {
        return __awaiter(this, void 0, void 0, function* () {
            if (this.videoMetadata === undefined || this.poses === undefined)
                return '{}';
            if (!this.isFinalized) {
                yield this.finalize();
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
                    // BodyVector の圧縮
                    const bodyVector = [];
                    for (const key of PoseSet.BODY_VECTOR_MAPPINGS) {
                        bodyVector.push(pose.bodyVector[key]);
                    }
                    // HandVector の圧縮
                    let handVector = undefined;
                    if (pose.handVector) {
                        handVector = [];
                        for (const key of PoseSet.HAND_VECTOR_MAPPINGS) {
                            handVector.push(pose.handVector[key]);
                        }
                    }
                    // PoseSetJsonItem の pose オブジェクトを生成
                    return {
                        t: pose.timeMiliseconds,
                        d: pose.durationMiliseconds,
                        p: pose.pose,
                        l: pose.leftHand,
                        r: pose.rightHand,
                        v: bodyVector,
                        h: handVector,
                        e: pose.extendedData,
                        md: pose.mergedDurationMiliseconds,
                        mt: pose.mergedTimeMiliseconds,
                    };
                }),
                poseLandmarkMapppings: poseLandmarkMappings,
            };
            return JSON.stringify(json);
        });
    }
    /**
     * JSON からの読み込み
     * @param json JSON 文字列 または JSON オブジェクト
     */
    loadJson(json) {
        const parsedJson = typeof json === 'string' ? JSON.parse(json) : json;
        if (parsedJson.generator !== 'mp-video-pose-extractor') {
            throw '不正なファイル';
        }
        else if (parsedJson.version !== 1) {
            throw '未対応のバージョン';
        }
        this.videoMetadata = parsedJson.video;
        this.poses = parsedJson.poses.map((item) => {
            const bodyVector = {};
            PoseSet.BODY_VECTOR_MAPPINGS.map((key, index) => {
                bodyVector[key] = item.v[index];
            });
            const handVector = {};
            if (item.h) {
                PoseSet.HAND_VECTOR_MAPPINGS.map((key, index) => {
                    handVector[key] = item.h[index];
                });
            }
            return {
                timeMiliseconds: item.t,
                durationMiliseconds: item.d,
                pose: item.p,
                leftHand: item.l,
                rightHand: item.r,
                bodyVector: bodyVector,
                handVector: handVector,
                frameImageDataUrl: undefined,
                extendedData: item.e,
                debug: undefined,
                mergedDurationMiliseconds: item.md,
                mergedTimeMiliseconds: item.mt,
            };
        });
    }
    /**
     * ZIP ファイルからの読み込み
     * @param buffer ZIP ファイルの Buffer
     * @param includeImages 画像を展開するかどうか
     */
    loadZip(buffer, includeImages = true) {
        var _a, _b, _c;
        return __awaiter(this, void 0, void 0, function* () {
            const jsZip = new JSZip();
            console.debug(`[PoseSet] init...`);
            const zip = yield jsZip.loadAsync(buffer, { base64: false });
            if (!zip)
                throw 'ZIPファイルを読み込めませんでした';
            const json = yield ((_a = zip.file('poses.json')) === null || _a === void 0 ? void 0 : _a.async('text'));
            if (json === undefined) {
                throw 'ZIPファイルに pose.json が含まれていません';
            }
            this.loadJson(json);
            const fileExt = this.getFileExtensionByMime(this.IMAGE_MIME);
            if (includeImages) {
                for (const pose of this.poses) {
                    if (!pose.frameImageDataUrl) {
                        const frameImageFileName = `frame-${pose.timeMiliseconds}.${fileExt}`;
                        const imageBase64 = yield ((_b = zip
                            .file(frameImageFileName)) === null || _b === void 0 ? void 0 : _b.async('base64'));
                        if (imageBase64) {
                            pose.frameImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                        }
                    }
                    if (!pose.poseImageDataUrl) {
                        const poseImageFileName = `pose-${pose.timeMiliseconds}.${fileExt}`;
                        const imageBase64 = yield ((_c = zip
                            .file(poseImageFileName)) === null || _c === void 0 ? void 0 : _c.async('base64'));
                        if (imageBase64) {
                            pose.poseImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                        }
                    }
                }
            }
        });
    }
    pushPoseFromSimilarPoseQueue(nextPoseTimeMiliseconds) {
        if (this.similarPoseQueue.length === 0)
            return;
        if (this.similarPoseQueue.length === 1) {
            // 類似ポーズキューにポーズが一つしかない場合、当該ポーズをポーズ配列へ追加
            const pose = this.similarPoseQueue[0];
            this.poses.push(pose);
            this.similarPoseQueue = [];
            return;
        }
        // 各ポーズの持続時間を設定
        for (let i = 0; i < this.similarPoseQueue.length - 1; i++) {
            this.similarPoseQueue[i].durationMiliseconds =
                this.similarPoseQueue[i + 1].timeMiliseconds -
                    this.similarPoseQueue[i].timeMiliseconds;
        }
        if (nextPoseTimeMiliseconds) {
            this.similarPoseQueue[this.similarPoseQueue.length - 1].durationMiliseconds =
                nextPoseTimeMiliseconds -
                    this.similarPoseQueue[this.similarPoseQueue.length - 1].timeMiliseconds;
        }
        // 類似ポーズキューの中から最も持続時間が長いポーズを選択
        const selectedPose = PoseSet.getSuitablePoseByPoses(this.similarPoseQueue);
        // 選択されなかったポーズを列挙
        selectedPose.debug.duplicatedItems = this.similarPoseQueue
            .filter((item) => {
            return item.timeMiliseconds !== selectedPose.timeMiliseconds;
        })
            .map((item) => {
            return {
                timeMiliseconds: item.timeMiliseconds,
                durationMiliseconds: item.durationMiliseconds,
                bodySimilarity: undefined,
                handSimilarity: undefined,
            };
        });
        selectedPose.mergedTimeMiliseconds =
            this.similarPoseQueue[0].timeMiliseconds;
        selectedPose.mergedDurationMiliseconds = this.similarPoseQueue.reduce((sum, item) => {
            return sum + item.durationMiliseconds;
        }, 0);
        // 当該ポーズをポーズ配列へ追加
        if (this.IS_ENABLED_REMOVE_DUPLICATED_POSES_FOR_AROUND) {
            this.poses.push(selectedPose);
        }
        else {
            // デバッグ用
            this.poses.push(...this.similarPoseQueue);
        }
        // 類似ポーズキューをクリア
        this.similarPoseQueue = [];
    }
    removeDuplicatedPoses() {
        // 全ポーズを比較して類似ポーズを削除
        const newPoses = [], removedPoses = [];
        for (const pose of this.poses) {
            let duplicatedPose;
            for (const insertedPose of newPoses) {
                const isSimilarBodyPose = PoseSet.isSimilarBodyPose(pose.bodyVector, insertedPose.bodyVector);
                const isSimilarHandPose = pose.handVector && insertedPose.handVector
                    ? PoseSet.isSimilarHandPose(pose.handVector, insertedPose.handVector, 0.9)
                    : false;
                if (isSimilarBodyPose && isSimilarHandPose) {
                    // 身体・手ともに類似ポーズならば
                    duplicatedPose = insertedPose;
                    break;
                }
            }
            if (duplicatedPose) {
                removedPoses.push(pose);
                if (duplicatedPose.debug.duplicatedItems) {
                    duplicatedPose.debug.duplicatedItems.push({
                        timeMiliseconds: pose.timeMiliseconds,
                        durationMiliseconds: pose.durationMiliseconds,
                        bodySimilarity: undefined,
                        handSimilarity: undefined,
                    });
                }
                continue;
            }
            newPoses.push(pose);
        }
        console.info(`[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`, {
            removed: removedPoses,
            keeped: newPoses,
        });
        this.poses = newPoses;
    }
    getFileExtensionByMime(IMAGE_MIME) {
        switch (IMAGE_MIME) {
            case 'image/png':
                return 'png';
            case 'image/jpeg':
                return 'jpg';
            case 'image/webp':
                return 'webp';
            default:
                return 'png';
        }
    }
}
// BodyVector のキー名
PoseSet.BODY_VECTOR_MAPPINGS = [
    // 右腕
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    // 左腕
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
];
// HandVector のキー名
PoseSet.HAND_VECTOR_MAPPINGS = [
    // 右手 - 親指
    'rightThumbTipToFirstJoint',
    'rightThumbFirstJointToSecondJoint',
    // 右手 - 人差し指
    'rightIndexFingerTipToFirstJoint',
    'rightIndexFingerFirstJointToSecondJoint',
    // 右手 - 中指
    'rightMiddleFingerTipToFirstJoint',
    'rightMiddleFingerFirstJointToSecondJoint',
    // 右手 - 薬指
    'rightRingFingerTipToFirstJoint',
    'rightRingFingerFirstJointToSecondJoint',
    // 右手 - 小指
    'rightPinkyFingerTipToFirstJoint',
    'rightPinkyFingerFirstJointToSecondJoint',
    // 左手 - 親指
    'leftThumbTipToFirstJoint',
    'leftThumbFirstJointToSecondJoint',
    // 左手 - 人差し指
    'leftIndexFingerTipToFirstJoint',
    'leftIndexFingerFirstJointToSecondJoint',
    // 左手 - 中指
    'leftMiddleFingerTipToFirstJoint',
    'leftMiddleFingerFirstJointToSecondJoint',
    // 左手 - 薬指
    'leftRingFingerTipToFirstJoint',
    'leftRingFingerFirstJointToSecondJoint',
    // 左手 - 小指
    'leftPinkyFingerTipToFirstJoint',
    'leftPinkyFingerFirstJointToSecondJoint',
];

/**
 * ポーズを管理するためのサービス
 */
class PoseComposerService {
    constructor() { }
    init(videoName) {
        const poseSet = new PoseSet();
        poseSet.setVideoName(videoName);
        return poseSet;
    }
    downloadAsJson(poseSet) {
        return __awaiter(this, void 0, void 0, function* () {
            const blob = new Blob([yield poseSet.getJson()], {
                type: 'application/json',
            });
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.target = '_blank';
            a.download = `${poseSet.getVideoName()}-poses.json`;
            a.click();
        });
    }
    downloadAsZip(poseSet) {
        return __awaiter(this, void 0, void 0, function* () {
            const content = yield poseSet.getZip();
            const url = window.URL.createObjectURL(content);
            const a = document.createElement('a');
            a.href = url;
            a.target = '_blank';
            a.download = `${poseSet.getVideoName()}-poses.zip`;
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
    getFacePreviewMediaStream() {
        if (!this.facePreviewCanvasElement)
            return;
        return this.facePreviewCanvasElement.captureStream();
    }
    onVideoFrame(input) {
        return __awaiter(this, void 0, void 0, function* () {
            if (!this.holistic)
                return;
            if (input instanceof HTMLVideoElement) {
                if (this.posePreviewCanvasElement) {
                    this.posePreviewCanvasElement.width = input.videoWidth;
                    this.posePreviewCanvasElement.height = input.videoHeight;
                }
                if (this.handPreviewCanvasElement) {
                    this.handPreviewCanvasElement.width = input.videoWidth;
                    this.handPreviewCanvasElement.height = input.videoHeight;
                }
                if (this.facePreviewCanvasElement) {
                    this.facePreviewCanvasElement.width = input.videoWidth;
                    this.facePreviewCanvasElement.height = input.videoHeight;
                }
            }
            else if (input instanceof HTMLCanvasElement) {
                if (this.posePreviewCanvasElement) {
                    this.posePreviewCanvasElement.width = input.width;
                    this.posePreviewCanvasElement.height = input.height;
                }
                if (this.handPreviewCanvasElement) {
                    this.handPreviewCanvasElement.width = input.width;
                    this.handPreviewCanvasElement.height = input.height;
                }
                if (this.facePreviewCanvasElement) {
                    this.facePreviewCanvasElement.width = input.width;
                    this.facePreviewCanvasElement.height = input.height;
                }
            }
            yield this.holistic.send({ image: input });
        });
    }
    init() {
        this.posePreviewCanvasElement = document.createElement('canvas');
        this.posePreviewCanvasContext =
            this.posePreviewCanvasElement.getContext('2d') || undefined;
        this.handPreviewCanvasElement = document.createElement('canvas');
        this.handPreviewCanvasContext =
            this.handPreviewCanvasElement.getContext('2d') || undefined;
        this.facePreviewCanvasElement = document.createElement('canvas');
        this.facePreviewCanvasContext =
            this.facePreviewCanvasElement.getContext('2d') || undefined;
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
        // 検出に使用したフレーム画像を保持 (加工されていない画像)
        const sourceImageDataUrl = this.posePreviewCanvasElement.toDataURL('image/png');
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
        // 手の領域のみのプレビュー画像を生成
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
        // 顔の領域のみのフレーム画像を生成
        if (this.facePreviewCanvasContext && this.facePreviewCanvasElement) {
            const FACE_PREVIEW_ZOOM = 1.25;
            const facePreviewBaseY = this.facePreviewCanvasElement.height / 2;
            if (results.faceLandmarks) {
                const rect = this.getRectByLandmarks(results.faceLandmarks, results.image.width, results.image.height);
                const rectWidth = rect[2] * FACE_PREVIEW_ZOOM;
                const rectHeight = rect[3] * FACE_PREVIEW_ZOOM;
                this.facePreviewCanvasElement.width = rectWidth;
                this.facePreviewCanvasElement.height = rectHeight;
                this.facePreviewCanvasContext.clearRect(0, 0, rectWidth, rectHeight);
                this.facePreviewCanvasContext.drawImage(results.image, rect[0] - 10, rect[1] - 10, rect[2] + 10, rect[3] + 10, 0, 0, rectWidth, rectHeight);
            }
            else {
                this.facePreviewCanvasContext.clearRect(0, 0, this.facePreviewCanvasElement.width, this.facePreviewCanvasElement.height);
            }
        }
        // イベントを送出
        this.onResultsEventEmitter.emit({
            mpResults: results,
            // 加工されていない画像 (PNG)
            frameImageDataUrl: sourceImageDataUrl,
            // 加工された画像 (PNG)
            posePreviewImageDataUrl: this.posePreviewCanvasElement.toDataURL('image/png'),
            // 顔のみの画像 (PNG)
            faceFrameImageDataUrl: results.faceLandmarks
                ? this.facePreviewCanvasElement.toDataURL('image/png')
                : undefined,
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

export { NgxMpPoseExtractorComponent, NgxMpPoseExtractorModule, NgxMpPoseExtractorService, PoseComposerService, PoseExtractorService, PoseSet };
//# sourceMappingURL=ngx-mp-pose-extractor.mjs.map
