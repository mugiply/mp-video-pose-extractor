import { POSE_LANDMARKS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { ImageTrimmer } from './internals/image-trimmer';
export class PoseSet {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        // 画像書き出し時の設定
        this.IMAGE_WIDTH = 1080;
        this.IMAGE_MIME = 'image/webp';
        this.IMAGE_QUALITY = 0.7;
        // 画像の背景色置換
        this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR = '#016AFD';
        this.IMAGE_BACKGROUND_REPLACE_DST_COLOR = '#FFFFFF00';
        this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD = 100;
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
    getPoseByTime(timeMiliseconds) {
        if (this.poses === undefined)
            return undefined;
        return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
    }
    pushPose(videoTimeMiliseconds, frameImageDataUrl, poseImageDataUrl, videoWidth, videoHeight, videoDuration, results) {
        this.setVideoMetaData(videoWidth, videoHeight, videoDuration);
        if (results.poseLandmarks === undefined)
            return;
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
            pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
                return [landmark.x, landmark.y, landmark.z, landmark.visibility];
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
        // 全ポーズを比較して類似ポーズを削除
        if (PoseSet.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
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
            console.info(`[PoseSet] getJson - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`);
            this.poses = newPoses;
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
        // 画像を整形
        for (const pose of this.poses) {
            let imageTrimmer = new ImageTrimmer();
            if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
                continue;
            }
            // 画像を整形 - フレーム画像
            console.log(`[PoseSet] finalize - Processing frame image...`, pose.timeMiliseconds);
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            const marginColor = await imageTrimmer.getMarginColor();
            console.log(`[PoseSet] finalize - Detected margin color...`, pose.timeMiliseconds, marginColor);
            if (marginColor === null)
                continue;
            if (marginColor !== '#000000') {
                console.warn(`[PoseSet] finalize - Skip this frame image, because the margin color is not black.`);
                continue;
            }
            const trimmed = await imageTrimmer.trimMargin(marginColor);
            console.log(`[PoseSet] finalize - Trimmed margin of frame image...`, pose.timeMiliseconds, trimmed);
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
            await imageTrimmer.crop(0, trimmed.marginTop, trimmed.width, trimmed.heightNew);
            console.log(`[PoseSet] finalize - Trimmed margin of pose preview image...`, pose.timeMiliseconds, trimmed);
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
        this.poses = parsedJson.poses.map((item) => {
            const poseVector = {};
            PoseSet.POSE_VECTOR_MAPPINGS.map((key, index) => {
                poseVector[key] = item.vectors[index];
            });
            return {
                timeMiliseconds: item.t,
                durationMiliseconds: item.d,
                pose: item.pose,
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
PoseSet.IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;
PoseSet.POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBRXpELE1BQU0sT0FBTyxPQUFPO0lBZ0NsQjtRQXZCTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQVdyQyxhQUFhO1FBQ0ksZ0JBQVcsR0FBVyxJQUFJLENBQUM7UUFDM0IsZUFBVSxHQUN6QixZQUFZLENBQUM7UUFDRSxrQkFBYSxHQUFHLEdBQUcsQ0FBQztRQUVyQyxXQUFXO1FBQ00sdUNBQWtDLEdBQUcsU0FBUyxDQUFDO1FBQy9DLHVDQUFrQyxHQUFHLFdBQVcsQ0FBQztRQUNqRCw0Q0FBdUMsR0FBRyxHQUFHLENBQUM7UUFHN0QsSUFBSSxDQUFDLGFBQWEsR0FBRztZQUNuQixJQUFJLEVBQUUsRUFBRTtZQUNSLEtBQUssRUFBRSxDQUFDO1lBQ1IsTUFBTSxFQUFFLENBQUM7WUFDVCxRQUFRLEVBQUUsQ0FBQztTQUNaLENBQUM7SUFDSixDQUFDO0lBRUQsWUFBWTtRQUNWLE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUM7SUFDakMsQ0FBQztJQUVELFlBQVksQ0FBQyxTQUFpQjtRQUM1QixJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksR0FBRyxTQUFTLENBQUM7SUFDdEMsQ0FBQztJQUVELGdCQUFnQixDQUFDLEtBQWEsRUFBRSxNQUFjLEVBQUUsUUFBZ0I7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QUNwQixJQUFJLFdBQVcsRUFBRTt3QkFDZixJQUFJLENBQUMsZ0JBQWdCLEdBQUcsUUFBUSxJQUFJLENBQUMsVUFBVSxXQUFXLFdBQVcsRUFBRSxDQUFDO3FCQUN6RTtpQkFDRjthQUNGO1NBQ0Y7SUFDSCxDQUFDOztBQXBnQnNCLDJDQUFtQyxHQUFHLElBQUksQ0FBQztBQUUzQyw0QkFBb0IsR0FBRztJQUM1Qyx3QkFBd0I7SUFDeEIsMkJBQTJCO0lBQzNCLHNCQUFzQjtJQUN0Qix5QkFBeUI7Q0FDMUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IFBPU0VfTEFORE1BUktTLCBSZXN1bHRzIH0gZnJvbSAnQG1lZGlhcGlwZS9ob2xpc3RpYyc7XG5pbXBvcnQgKiBhcyBKU1ppcCBmcm9tICdqc3ppcCc7XG5pbXBvcnQgeyBQb3NlU2V0SXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtaXRlbSc7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbiB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtanNvbic7XG5pbXBvcnQgeyBQb3NlU2V0SnNvbkl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24taXRlbSc7XG5pbXBvcnQgeyBQb3NlVmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXZlY3Rvcic7XG5cbi8vIEB0cy1pZ25vcmVcbmltcG9ydCBjb3NTaW1pbGFyaXR5IGZyb20gJ2Nvcy1zaW1pbGFyaXR5JztcbmltcG9ydCB7IFNpbWlsYXJQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvbWF0Y2hlZC1wb3NlLWl0ZW0nO1xuaW1wb3J0IHsgSW1hZ2VUcmltbWVyIH0gZnJvbSAnLi9pbnRlcm5hbHMvaW1hZ2UtdHJpbW1lcic7XG5cbmV4cG9ydCBjbGFzcyBQb3NlU2V0IHtcbiAgcHVibGljIGdlbmVyYXRvcj86IHN0cmluZztcbiAgcHVibGljIHZlcnNpb24/OiBudW1iZXI7XG4gIHByaXZhdGUgdmlkZW9NZXRhZGF0YSE6IHtcbiAgICBuYW1lOiBzdHJpbmc7XG4gICAgd2lkdGg6IG51bWJlcjtcbiAgICBoZWlnaHQ6IG51bWJlcjtcbiAgICBkdXJhdGlvbjogbnVtYmVyO1xuICB9O1xuICBwdWJsaWMgcG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgcHVibGljIGlzRmluYWxpemVkPzogYm9vbGVhbiA9IGZhbHNlO1xuXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSVNfRU5BQkxFX0RVUExJQ0FURURfUE9TRV9SRURVQ1RJT04gPSB0cnVlO1xuXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgUE9TRV9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgJ3JpZ2h0V3Jpc3RUb1JpZ2h0RWxib3cnLFxuICAgICdyaWdodEVsYm93VG9SaWdodFNob3VsZGVyJyxcbiAgICAnbGVmdFdyaXN0VG9MZWZ0RWxib3cnLFxuICAgICdsZWZ0RWxib3dUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgLy8g55S75YOP5pu444GN5Ye644GX5pmC44Gu6Kit5a6aXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEg6IG51bWJlciA9IDEwODA7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUlNRTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPVxuICAgICdpbWFnZS93ZWJwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9RVUFMSVRZID0gMC43O1xuXG4gIC8vIOeUu+WDj+OBruiDjOaZr+iJsue9ruaPm1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IgPSAnIzAxNkFGRCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUiA9ICcjRkZGRkZGMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRCA9IDEwMDtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSB7XG4gICAgICBuYW1lOiAnJyxcbiAgICAgIHdpZHRoOiAwLFxuICAgICAgaGVpZ2h0OiAwLFxuICAgICAgZHVyYXRpb246IDAsXG4gICAgfTtcbiAgfVxuXG4gIGdldFZpZGVvTmFtZSgpIHtcbiAgICByZXR1cm4gdGhpcy52aWRlb01ldGFkYXRhLm5hbWU7XG4gIH1cblxuICBzZXRWaWRlb05hbWUodmlkZW9OYW1lOiBzdHJpbmcpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEubmFtZSA9IHZpZGVvTmFtZTtcbiAgfVxuXG4gIHNldFZpZGVvTWV0YURhdGEod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGR1cmF0aW9uOiBudW1iZXIpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEud2lkdGggPSB3aWR0aDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuaGVpZ2h0ID0gaGVpZ2h0O1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiA9IGR1cmF0aW9uO1xuICB9XG5cbiAgZ2V0TnVtYmVyT2ZQb3NlcygpOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiAtMTtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5sZW5ndGg7XG4gIH1cblxuICBnZXRQb3NlcygpOiBQb3NlU2V0SXRlbVtdIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gW107XG4gICAgcmV0dXJuIHRoaXMucG9zZXM7XG4gIH1cblxuICBnZXRQb3NlQnlUaW1lKHRpbWVNaWxpc2Vjb25kczogbnVtYmVyKTogUG9zZVNldEl0ZW0gfCB1bmRlZmluZWQge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMuZmluZCgocG9zZSkgPT4gcG9zZS50aW1lTWlsaXNlY29uZHMgPT09IHRpbWVNaWxpc2Vjb25kcyk7XG4gIH1cblxuICBwdXNoUG9zZShcbiAgICB2aWRlb1RpbWVNaWxpc2Vjb25kczogbnVtYmVyLFxuICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgcG9zZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHZpZGVvV2lkdGg6IG51bWJlcixcbiAgICB2aWRlb0hlaWdodDogbnVtYmVyLFxuICAgIHZpZGVvRHVyYXRpb246IG51bWJlcixcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICkge1xuICAgIHRoaXMuc2V0VmlkZW9NZXRhRGF0YSh2aWRlb1dpZHRoLCB2aWRlb0hlaWdodCwgdmlkZW9EdXJhdGlvbik7XG5cbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHJldHVybjtcblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgLSBDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHdpdGggdGhlIHdvcmxkIGNvb3JkaW5hdGVgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBQb3NlU2V0LmdldFBvc2VWZWN0b3IocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUpO1xuICAgIGlmICghcG9zZVZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlIC0gQ291bGQgbm90IGdldCB0aGUgcG9zZSB2ZWN0b3JgLFxuICAgICAgICBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZVxuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBwb3NlOiBQb3NlU2V0SXRlbSA9IHtcbiAgICAgIHRpbWVNaWxpc2Vjb25kczogdmlkZW9UaW1lTWlsaXNlY29uZHMsXG4gICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiAtMSxcbiAgICAgIHBvc2U6IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLm1hcCgobGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtsYW5kbWFyay54LCBsYW5kbWFyay55LCBsYW5kbWFyay56LCBsYW5kbWFyay52aXNpYmlsaXR5XTtcbiAgICAgIH0pLFxuICAgICAgdmVjdG9yczogcG9zZVZlY3RvcixcbiAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBmcmFtZUltYWdlRGF0YVVybCxcbiAgICAgIHBvc2VJbWFnZURhdGFVcmw6IHBvc2VJbWFnZURhdGFVcmwsXG4gICAgfTtcblxuICAgIGlmICgxIDw9IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcbiAgICAgIGlmIChQb3NlU2V0LmlzU2ltaWxhclBvc2UobGFzdFBvc2UudmVjdG9ycywgcG9zZS52ZWN0b3JzKSkge1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIC8vIOWJjeWbnuOBruODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICB2aWRlb1RpbWVNaWxpc2Vjb25kcyAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgdGhpcy5wb3Nlcy5wdXNoKHBvc2UpO1xuICB9XG5cbiAgYXN5bmMgZmluYWxpemUoKSB7XG4gICAgaWYgKDAgPT0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIHRoaXMuaXNGaW5hbGl6ZWQgPSB0cnVlO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIOWFqOODneODvOOCuuOCkuavlOi8g+OBl+OBpumhnuS8vOODneODvOOCuuOCkuWJiumZpFxuICAgIGlmIChQb3NlU2V0LklTX0VOQUJMRV9EVVBMSUNBVEVEX1BPU0VfUkVEVUNUSU9OKSB7XG4gICAgICBjb25zdCBuZXdQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICAgICAgZm9yIChjb25zdCBwb3NlQSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICAgIGxldCBpc0R1cGxpY2F0ZWQgPSBmYWxzZTtcbiAgICAgICAgZm9yIChjb25zdCBwb3NlQiBvZiBuZXdQb3Nlcykge1xuICAgICAgICAgIGlmIChQb3NlU2V0LmlzU2ltaWxhclBvc2UocG9zZUEudmVjdG9ycywgcG9zZUIudmVjdG9ycykpIHtcbiAgICAgICAgICAgIGlzRHVwbGljYXRlZCA9IHRydWU7XG4gICAgICAgICAgICBicmVhaztcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgaWYgKGlzRHVwbGljYXRlZCkgY29udGludWU7XG5cbiAgICAgICAgbmV3UG9zZXMucHVzaChwb3NlQSk7XG4gICAgICB9XG5cbiAgICAgIGNvbnNvbGUuaW5mbyhcbiAgICAgICAgYFtQb3NlU2V0XSBnZXRKc29uIC0gUmVkdWNlZCAke3RoaXMucG9zZXMubGVuZ3RofSBwb3NlcyAtPiAke25ld1Bvc2VzLmxlbmd0aH0gcG9zZXNgXG4gICAgICApO1xuICAgICAgdGhpcy5wb3NlcyA9IG5ld1Bvc2VzO1xuICAgIH1cblxuICAgIC8vIOacgOW+jOOBruODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgIGlmICgxIDw9IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcbiAgICAgIGlmIChsYXN0UG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzID09IC0xKSB7XG4gICAgICAgIGNvbnN0IHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gLSBsYXN0UG9zZS50aW1lTWlsaXNlY29uZHM7XG4gICAgICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgICBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcztcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDnlLvlg4/jgpLmlbTlvaJcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCB8fCAhcG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gUHJvY2Vzc2luZyBmcmFtZSBpbWFnZS4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApO1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGNvbnN0IG1hcmdpbkNvbG9yID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldE1hcmdpbkNvbG9yKCk7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGVkIG1hcmdpbiBjb2xvci4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICBtYXJnaW5Db2xvclxuICAgICAgKTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciA9PT0gbnVsbCkgY29udGludWU7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgIT09ICcjMDAwMDAwJykge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIFNraXAgdGhpcyBmcmFtZSBpbWFnZSwgYmVjYXVzZSB0aGUgbWFyZ2luIGNvbG9yIGlzIG5vdCBibGFjay5gXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICBjb25zdCB0cmltbWVkID0gYXdhaXQgaW1hZ2VUcmltbWVyLnRyaW1NYXJnaW4obWFyZ2luQ29sb3IpO1xuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBUcmltbWVkIG1hcmdpbiBvZiBmcmFtZSBpbWFnZS4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICB0cmltbWVkXG4gICAgICApO1xuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVwbGFjZUNvbG9yKFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbGV0IG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZnJhbWUgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODneODvOOCuuODl+ODrOODk+ODpeODvOeUu+WDj1xuICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgIDAsXG4gICAgICAgIHRyaW1tZWQubWFyZ2luVG9wLFxuICAgICAgICB0cmltbWVkLndpZHRoLFxuICAgICAgICB0cmltbWVkLmhlaWdodE5ld1xuICAgICAgKTtcbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gVHJpbW1lZCBtYXJnaW4gb2YgcG9zZSBwcmV2aWV3IGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgIHRyaW1tZWRcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBwb3NlIHByZXZpZXcgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcbiAgICB9XG5cbiAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgfVxuXG4gIGdldFNpbWlsYXJQb3NlcyhcbiAgICByZXN1bHRzOiBSZXN1bHRzLFxuICAgIHRocmVzaG9sZDogbnVtYmVyID0gMC45XG4gICk6IFNpbWlsYXJQb3NlSXRlbVtdIHtcbiAgICBjb25zdCBwb3NlVmVjdG9yID0gUG9zZVNldC5nZXRQb3NlVmVjdG9yKChyZXN1bHRzIGFzIGFueSkuZWEpO1xuICAgIGlmICghcG9zZVZlY3RvcikgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIHBvc2UgdmVjdG9yJztcblxuICAgIGNvbnN0IHBvc2VzID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldFBvc2VTaW1pbGFyaXR5KHBvc2UudmVjdG9ycywgcG9zZVZlY3Rvcik7XG4gICAgICBpZiAodGhyZXNob2xkIDw9IHNpbWlsYXJpdHkpIHtcbiAgICAgICAgcG9zZXMucHVzaCh7XG4gICAgICAgICAgLi4ucG9zZSxcbiAgICAgICAgICBzaW1pbGFyaXR5OiBzaW1pbGFyaXR5LFxuICAgICAgICB9KTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gcG9zZXM7XG4gIH1cblxuICBzdGF0aWMgZ2V0UG9zZVZlY3RvcihcbiAgICBwb3NlTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdXG4gICk6IFBvc2VWZWN0b3IgfCB1bmRlZmluZWQge1xuICAgIHJldHVybiB7XG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIHN0YXRpYyBpc1NpbWlsYXJQb3NlKFxuICAgIHBvc2VWZWN0b3JBOiBQb3NlVmVjdG9yLFxuICAgIHBvc2VWZWN0b3JCOiBQb3NlVmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuOVxuICApOiBib29sZWFuIHtcbiAgICBsZXQgaXNTaW1pbGFyID0gZmFsc2U7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0UG9zZVNpbWlsYXJpdHkocG9zZVZlY3RvckEsIHBvc2VWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQpIGlzU2ltaWxhciA9IHRydWU7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGlzU2ltaWxhclBvc2VgLCBpc1NpbWlsYXIsIHNpbWlsYXJpdHkpO1xuXG4gICAgcmV0dXJuIGlzU2ltaWxhcjtcbiAgfVxuXG4gIHN0YXRpYyBnZXRQb3NlU2ltaWxhcml0eShcbiAgICBwb3NlVmVjdG9yQTogUG9zZVZlY3RvcixcbiAgICBwb3NlVmVjdG9yQjogUG9zZVZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllcyA9IHtcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBwb3NlVmVjdG9yQS5sZWZ0V3Jpc3RUb0xlZnRFbGJvdyxcbiAgICAgICAgcG9zZVZlY3RvckIubGVmdFdyaXN0VG9MZWZ0RWxib3dcbiAgICAgICksXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgcG9zZVZlY3RvckEubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXIsXG4gICAgICAgIHBvc2VWZWN0b3JCLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyXG4gICAgICApLFxuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgcG9zZVZlY3RvckEucmlnaHRXcmlzdFRvUmlnaHRFbGJvdyxcbiAgICAgICAgcG9zZVZlY3RvckIucmlnaHRXcmlzdFRvUmlnaHRFbGJvd1xuICAgICAgKSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIHBvc2VWZWN0b3JBLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXIsXG4gICAgICAgIHBvc2VWZWN0b3JCLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXJcbiAgICAgICksXG4gICAgfTtcblxuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1N1bSA9IE9iamVjdC52YWx1ZXMoY29zU2ltaWxhcml0aWVzKS5yZWR1Y2UoXG4gICAgICAoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsXG4gICAgICAwXG4gICAgKTtcbiAgICByZXR1cm4gY29zU2ltaWxhcml0aWVzU3VtIC8gT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzKS5sZW5ndGg7XG4gIH1cblxuICBwdWJsaWMgYXN5bmMgZ2V0WmlwKCk6IFByb21pc2U8QmxvYj4ge1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAganNaaXAuZmlsZSgncG9zZXMuanNvbicsIGF3YWl0IHRoaXMuZ2V0SnNvbigpKTtcblxuICAgIGNvbnN0IGltYWdlRmlsZUV4dCA9IHRoaXMuZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZSh0aGlzLklNQUdFX01JTUUpO1xuXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLnBvc2VJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBwb3NlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBhd2FpdCBqc1ppcC5nZW5lcmF0ZUFzeW5jKHsgdHlwZTogJ2Jsb2InIH0pO1xuICB9XG5cbiAgZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZShJTUFHRV9NSU1FOiBzdHJpbmcpIHtcbiAgICBzd2l0Y2ggKElNQUdFX01JTUUpIHtcbiAgICAgIGNhc2UgJ2ltYWdlL3BuZyc6XG4gICAgICAgIHJldHVybiAncG5nJztcbiAgICAgIGNhc2UgJ2ltYWdlL2pwZWcnOlxuICAgICAgICByZXR1cm4gJ2pwZyc7XG4gICAgICBjYXNlICdpbWFnZS93ZWJwJzpcbiAgICAgICAgcmV0dXJuICd3ZWJwJztcbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIHJldHVybiAncG5nJztcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgYXN5bmMgZ2V0SnNvbigpOiBQcm9taXNlPHN0cmluZz4ge1xuICAgIGlmICh0aGlzLnZpZGVvTWV0YWRhdGEgPT09IHVuZGVmaW5lZCB8fCB0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpXG4gICAgICByZXR1cm4gJ3t9JztcblxuICAgIGlmICghdGhpcy5pc0ZpbmFsaXplZCkge1xuICAgICAgYXdhaXQgdGhpcy5maW5hbGl6ZSgpO1xuICAgIH1cblxuICAgIGxldCBwb3NlTGFuZG1hcmtNYXBwaW5ncyA9IFtdO1xuICAgIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKFBPU0VfTEFORE1BUktTKSkge1xuICAgICAgY29uc3QgaW5kZXg6IG51bWJlciA9IFBPU0VfTEFORE1BUktTW2tleSBhcyBrZXlvZiB0eXBlb2YgUE9TRV9MQU5ETUFSS1NdO1xuICAgICAgcG9zZUxhbmRtYXJrTWFwcGluZ3NbaW5kZXhdID0ga2V5O1xuICAgIH1cblxuICAgIGNvbnN0IGpzb246IFBvc2VTZXRKc29uID0ge1xuICAgICAgZ2VuZXJhdG9yOiAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InLFxuICAgICAgdmVyc2lvbjogMSxcbiAgICAgIHZpZGVvOiB0aGlzLnZpZGVvTWV0YWRhdGEhLFxuICAgICAgcG9zZXM6IHRoaXMucG9zZXMubWFwKChwb3NlOiBQb3NlU2V0SXRlbSk6IFBvc2VTZXRKc29uSXRlbSA9PiB7XG4gICAgICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBbXTtcbiAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZVNldC5QT1NFX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgIHBvc2VWZWN0b3IucHVzaChwb3NlLnZlY3RvcnNba2V5IGFzIGtleW9mIFBvc2VWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdDogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgZDogcG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIHBvc2U6IHBvc2UucG9zZSxcbiAgICAgICAgICB2ZWN0b3JzOiBwb3NlVmVjdG9yLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcCgoaXRlbTogUG9zZVNldEpzb25JdGVtKTogUG9zZVNldEl0ZW0gPT4ge1xuICAgICAgY29uc3QgcG9zZVZlY3RvcjogYW55ID0ge307XG4gICAgICBQb3NlU2V0LlBPU0VfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICBwb3NlVmVjdG9yW2tleSBhcyBrZXlvZiBQb3NlVmVjdG9yXSA9IGl0ZW0udmVjdG9yc1tpbmRleF07XG4gICAgICB9KTtcblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgdGltZU1pbGlzZWNvbmRzOiBpdGVtLnQsXG4gICAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IGl0ZW0uZCxcbiAgICAgICAgcG9zZTogaXRlbS5wb3NlLFxuICAgICAgICB2ZWN0b3JzOiBwb3NlVmVjdG9yLFxuICAgICAgICBmcmFtZUltYWdlRGF0YVVybDogdW5kZWZpbmVkLFxuICAgICAgfTtcbiAgICB9KTtcbiAgfVxuXG4gIGFzeW5jIGxvYWRaaXAoYnVmZmVyOiBBcnJheUJ1ZmZlciwgaW5jbHVkZUltYWdlczogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGxvYWRaaXAuLi5gLCBKU1ppcCk7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGluaXQuLi5gKTtcbiAgICBjb25zdCB6aXAgPSBhd2FpdCBqc1ppcC5sb2FkQXN5bmMoYnVmZmVyLCB7IGJhc2U2NDogZmFsc2UgfSk7XG4gICAgaWYgKCF6aXApIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgpLoqq3jgb/ovrzjgoHjgb7jgZvjgpPjgafjgZfjgZ8nO1xuXG4gICAgY29uc3QganNvbiA9IGF3YWl0IHppcC5maWxlKCdwb3Nlcy5qc29uJyk/LmFzeW5jKCd0ZXh0Jyk7XG4gICAgaWYgKGpzb24gPT09IHVuZGVmaW5lZCkge1xuICAgICAgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OBqyBwb3NlLmpzb24g44GM5ZCr44G+44KM44Gm44GE44G+44Gb44KTJztcbiAgICB9XG5cbiAgICB0aGlzLmxvYWRKc29uKGpzb24pO1xuXG4gICAgY29uc3QgZmlsZUV4dCA9IHRoaXMuZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZSh0aGlzLklNQUdFX01JTUUpO1xuXG4gICAgaWYgKGluY2x1ZGVJbWFnZXMpIHtcbiAgICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IGZyYW1lSW1hZ2VGaWxlTmFtZSA9IGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ZpbGVFeHR9YDtcbiAgICAgICAgICBjb25zdCBpbWFnZUJhc2U2NCA9IGF3YWl0IHppcFxuICAgICAgICAgICAgLmZpbGUoZnJhbWVJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsID0gYGRhdGE6JHt0aGlzLklNQUdFX01JTUV9O2Jhc2U2NCwke2ltYWdlQmFzZTY0fWA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGlmICghcG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc3QgcG9zZUltYWdlRmlsZU5hbWUgPSBgcG9zZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ZpbGVFeHR9YDtcbiAgICAgICAgICBjb25zdCBpbWFnZUJhc2U2NCA9IGF3YWl0IHppcFxuICAgICAgICAgICAgLmZpbGUocG9zZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxufVxuIl19