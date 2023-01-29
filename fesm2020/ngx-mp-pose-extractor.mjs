import * as i0 from '@angular/core';
import { Injectable, Component, NgModule, EventEmitter } from '@angular/core';
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
    async loadByDataUrl(dataUrl) {
        const image = await new Promise((resolve, reject) => {
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
    }
    async trimMargin(marginColor, diffThreshold = 10) {
        if (this.canvas === undefined)
            throw new Error('Image is not loaded');
        // マージンを検出する範囲を指定 (左端から0〜20%)
        const edgeDetectionRangeMinX = 0;
        const edgeDetectionRangeMaxX = Math.floor(this.canvas.width * 0.2);
        // マージンの端を検出
        const edgePositionFromTop = await this.getVerticalEdgePositionOfColor(marginColor, 'top', diffThreshold, edgeDetectionRangeMinX, edgeDetectionRangeMaxX);
        const marginTop = edgePositionFromTop != null ? edgePositionFromTop : 0;
        const edgePositionFromBottom = await this.getVerticalEdgePositionOfColor(marginColor, 'bottom', diffThreshold, edgeDetectionRangeMinX, edgeDetectionRangeMaxX);
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
    }
    async crop(x, y, w, h) {
        if (!this.canvas || !this.context)
            return;
        const newCanvas = document.createElement('canvas');
        const newContext = newCanvas.getContext('2d');
        newCanvas.width = w;
        newCanvas.height = h;
        newContext.drawImage(this.canvas, x, y, newCanvas.width, newCanvas.height, 0, 0, newCanvas.width, newCanvas.height);
        this.replaceCanvas(newCanvas);
    }
    async replaceColor(srcColor, dstColor, diffThreshold) {
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
    }
    async getMarginColor() {
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
    }
    async getVerticalEdgePositionOfColor(color, direction, diffThreshold, minX, maxX) {
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
    }
    async getWidth() {
        return this.canvas?.width;
    }
    async getHeight() {
        return this.canvas?.height;
    }
    async resizeWithFit(param) {
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
    async getDataUrl(mime = 'image/jpeg', imageQuality) {
        if (!this.canvas) {
            return null;
        }
        if (mime === 'image/jpeg' || mime === 'image/webp') {
            return this.canvas.toDataURL(mime, imageQuality);
        }
        else {
            return this.canvas.toDataURL(mime);
        }
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
    getPoseByTime(timeMiliseconds) {
        if (this.poses === undefined)
            return undefined;
        return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
    }
    pushPose(videoTimeMiliseconds, frameImageDataUrl, poseImageDataUrl, videoWidth, videoHeight, videoDuration, results) {
        this.setVideoMetaData(videoWidth, videoHeight, videoDuration);
        if (results.poseLandmarks === undefined)
            return;
        if (this.poses.length === 0) {
            this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
        }
        const poseLandmarksWithWorldCoordinate = results.ea
            ? results.ea
            : [];
        if (poseLandmarksWithWorldCoordinate.length === 0) {
            console.warn(`[PoseSet] pushPose - Could not get the pose with the world coordinate`, results);
            return;
        }
        const poseVector = PoseSet.getPoseVector(poseLandmarksWithWorldCoordinate);
        if (!poseVector) {
            console.warn(`[PoseSet] pushPose - Could not get the pose vector`, poseLandmarksWithWorldCoordinate);
            return;
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
            leftHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            rightHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            vectors: poseVector,
            frameImageDataUrl: frameImageDataUrl,
            poseImageDataUrl: poseImageDataUrl,
        };
        if (1 <= this.poses.length) {
            const lastPose = this.poses[this.poses.length - 1];
            if (PoseSet.isSimilarPose(lastPose.vectors, pose.vectors)) {
                return;
            }
            // 前回のポーズの持続時間を設定
            const poseDurationMiliseconds = videoTimeMiliseconds - lastPose.timeMiliseconds;
            this.poses[this.poses.length - 1].durationMiliseconds =
                poseDurationMiliseconds;
        }
        this.poses.push(pose);
    }
    async finalize() {
        if (0 == this.poses.length) {
            this.isFinalized = true;
            return;
        }
        // 最後のポーズの持続時間を設定
        if (1 <= this.poses.length) {
            const lastPose = this.poses[this.poses.length - 1];
            if (lastPose.durationMiliseconds == -1) {
                const poseDurationMiliseconds = this.videoMetadata.duration - lastPose.timeMiliseconds;
                this.poses[this.poses.length - 1].durationMiliseconds =
                    poseDurationMiliseconds;
            }
        }
        // 重複ポーズを除去
        this.removeDuplicatedPoses();
        // 画像のマージンを取得
        console.log(`[PoseSet] finalize - Detecting image margins...`);
        let imageTrimming = undefined;
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl) {
                continue;
            }
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            const marginColor = await imageTrimmer.getMarginColor();
            console.log(`[PoseSet] finalize - Detected margin color...`, pose.timeMiliseconds, marginColor);
            if (marginColor === null)
                continue;
            if (marginColor !== this.IMAGE_MARGIN_TRIMMING_COLOR) {
                continue;
            }
            const trimmed = await imageTrimmer.trimMargin(marginColor, this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD);
            if (!trimmed)
                continue;
            imageTrimming = trimmed;
            console.log(`[PoseSet] finalize - Determined image trimming positions...`, trimmed);
            break;
        }
        // 画像を整形
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
                continue;
            }
            console.log(`[PoseSet] finalize - Processing image...`, pose.timeMiliseconds);
            // 画像を整形 - フレーム画像
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            if (imageTrimming) {
                await imageTrimmer.crop(0, imageTrimming.marginTop, imageTrimming.width, imageTrimming.heightNew);
            }
            await imageTrimmer.replaceColor(this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR, this.IMAGE_BACKGROUND_REPLACE_DST_COLOR, this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD);
            await imageTrimmer.resizeWithFit({
                width: this.IMAGE_WIDTH,
            });
            let newDataUrl = await imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                ? this.IMAGE_QUALITY
                : undefined);
            if (!newDataUrl) {
                console.warn(`[PoseSet] finalize - Could not get the new dataurl for frame image`);
                continue;
            }
            pose.frameImageDataUrl = newDataUrl;
            // 画像を整形 - ポーズプレビュー画像
            imageTrimmer = new ImageTrimmer();
            await imageTrimmer.loadByDataUrl(pose.poseImageDataUrl);
            if (imageTrimming) {
                await imageTrimmer.crop(0, imageTrimming.marginTop, imageTrimming.width, imageTrimming.heightNew);
            }
            await imageTrimmer.resizeWithFit({
                width: this.IMAGE_WIDTH,
            });
            newDataUrl = await imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
                ? this.IMAGE_QUALITY
                : undefined);
            if (!newDataUrl) {
                console.warn(`[PoseSet] finalize - Could not get the new dataurl for pose preview image`);
                continue;
            }
            pose.poseImageDataUrl = newDataUrl;
        }
        this.isFinalized = true;
    }
    removeDuplicatedPoses() {
        // 全ポーズを比較して類似ポーズを削除
        const newPoses = [];
        for (const poseA of this.poses) {
            let isDuplicated = false;
            for (const poseB of newPoses) {
                if (PoseSet.isSimilarPose(poseA.vectors, poseB.vectors)) {
                    isDuplicated = true;
                    break;
                }
            }
            if (isDuplicated)
                continue;
            newPoses.push(poseA);
        }
        console.info(`[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`);
        this.poses = newPoses;
    }
    getSimilarPoses(results, threshold = 0.9) {
        const poseVector = PoseSet.getPoseVector(results.ea);
        if (!poseVector)
            throw 'Could not get the pose vector';
        const poses = [];
        for (const pose of this.poses) {
            const similarity = PoseSet.getPoseSimilarity(pose.vectors, poseVector);
            if (threshold <= similarity) {
                poses.push({
                    ...pose,
                    similarity: similarity,
                });
            }
        }
        return poses;
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
        let isSimilar = false;
        const similarity = PoseSet.getPoseSimilarity(poseVectorA, poseVectorB);
        if (similarity >= threshold)
            isSimilar = true;
        // console.log(`[PoseSet] isSimilarPose`, isSimilar, similarity);
        return isSimilar;
    }
    static getPoseSimilarity(poseVectorA, poseVectorB) {
        const cosSimilarities = {
            leftWristToLeftElbow: cosSimilarity(poseVectorA.leftWristToLeftElbow, poseVectorB.leftWristToLeftElbow),
            leftElbowToLeftShoulder: cosSimilarity(poseVectorA.leftElbowToLeftShoulder, poseVectorB.leftElbowToLeftShoulder),
            rightWristToRightElbow: cosSimilarity(poseVectorA.rightWristToRightElbow, poseVectorB.rightWristToRightElbow),
            rightElbowToRightShoulder: cosSimilarity(poseVectorA.rightElbowToRightShoulder, poseVectorB.rightElbowToRightShoulder),
        };
        const cosSimilaritiesSum = Object.values(cosSimilarities).reduce((sum, value) => sum + value, 0);
        return cosSimilaritiesSum / Object.keys(cosSimilarities).length;
    }
    async getZip() {
        const jsZip = new JSZip();
        jsZip.file('poses.json', await this.getJson());
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
        }
        return await jsZip.generateAsync({ type: 'blob' });
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
    async getJson() {
        if (this.videoMetadata === undefined || this.poses === undefined)
            return '{}';
        if (!this.isFinalized) {
            await this.finalize();
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
                for (const key of PoseSet.POSE_VECTOR_MAPPINGS) {
                    poseVector.push(pose.vectors[key]);
                }
                return {
                    t: pose.timeMiliseconds,
                    d: pose.durationMiliseconds,
                    p: pose.pose,
                    l: pose.leftHand,
                    r: pose.rightHand,
                    v: poseVector,
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
        this.poses = parsedJson.poses.map((item) => {
            const poseVector = {};
            PoseSet.POSE_VECTOR_MAPPINGS.map((key, index) => {
                poseVector[key] = item.v[index];
            });
            return {
                timeMiliseconds: item.t,
                durationMiliseconds: item.d,
                pose: item.p,
                leftHand: item.l,
                rightHand: item.r,
                vectors: poseVector,
                frameImageDataUrl: undefined,
            };
        });
    }
    async loadZip(buffer, includeImages = true) {
        console.log(`[PoseSet] loadZip...`, JSZip);
        const jsZip = new JSZip();
        console.log(`[PoseSet] init...`);
        const zip = await jsZip.loadAsync(buffer, { base64: false });
        if (!zip)
            throw 'ZIPファイルを読み込めませんでした';
        const json = await zip.file('poses.json')?.async('text');
        if (json === undefined) {
            throw 'ZIPファイルに pose.json が含まれていません';
        }
        this.loadJson(json);
        const fileExt = this.getFileExtensionByMime(this.IMAGE_MIME);
        if (includeImages) {
            for (const pose of this.poses) {
                if (!pose.frameImageDataUrl) {
                    const frameImageFileName = `frame-${pose.timeMiliseconds}.${fileExt}`;
                    const imageBase64 = await zip
                        .file(frameImageFileName)
                        ?.async('base64');
                    if (imageBase64) {
                        pose.frameImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                    }
                }
                if (!pose.poseImageDataUrl) {
                    const poseImageFileName = `pose-${pose.timeMiliseconds}.${fileExt}`;
                    const imageBase64 = await zip
                        .file(poseImageFileName)
                        ?.async('base64');
                    if (imageBase64) {
                        pose.poseImageDataUrl = `data:${this.IMAGE_MIME};base64,${imageBase64}`;
                    }
                }
            }
        }
    }
}
PoseSet.POSE_VECTOR_MAPPINGS = [
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
        const poseSet = new PoseSet();
        poseSet.setVideoName(videoName);
        return poseSet;
    }
    async downloadAsJson(poseSet) {
        const blob = new Blob([await poseSet.getJson()], {
            type: 'application/json',
        });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${poseSet.getVideoName()}-poses.json`;
        a.click();
    }
    async downloadAsZip(poseSet) {
        const content = await poseSet.getZip();
        const url = window.URL.createObjectURL(content);
        const a = document.createElement('a');
        a.href = url;
        a.target = '_blank';
        a.download = `${poseSet.getVideoName()}-poses.zip`;
        a.click();
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
            // 加工されていない画像 (PNG)
            sourceImageDataUrl: sourceImageDataUrl,
            // 加工された画像 (PNG)
            posePreviewImageDataUrl: this.posePreviewCanvasElement.toDataURL('image/png'),
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