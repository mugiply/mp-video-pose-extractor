import { POSE_LANDMARKS } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { ImageTrimmer } from './internals/image-trimmer';
export class Pose {
    constructor() {
        this.poses = [];
        this.isFinalized = false;
        this.IMAGE_JPEG_QUALITY = 0.7;
        this.IMAGE_WIDTH = 900;
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
            timeMiliseconds: videoTimeMiliseconds,
            durationMiliseconds: -1,
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
        if (Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
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
            console.log(`[Pose] finalize - Processing frame image...`, pose.timeMiliseconds);
            await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);
            const marginColor = await imageTrimmer.getMarginColor();
            console.log(`[Pose] finalize - Detected margin color...`, pose.timeMiliseconds, marginColor);
            if (marginColor === null)
                continue;
            if (marginColor !== '#000000') {
                console.warn(`[Pose] finalize - Skip this frame image, because the margin color is not black.`);
                continue;
            }
            const trimmed = await imageTrimmer.trimMargin(marginColor);
            console.log(`[Pose] finalize - Trimmed margin of frame image...`, pose.timeMiliseconds, trimmed);
            await imageTrimmer.resizeWithFit({
                width: this.IMAGE_WIDTH,
            });
            let newDataUrl = await imageTrimmer.getDataUrl('image/jpeg', this.IMAGE_JPEG_QUALITY);
            if (!newDataUrl) {
                console.warn(`[Pose] finalize - Could not get the new dataurl for frame image`);
                continue;
            }
            pose.frameImageDataUrl = newDataUrl;
            // 画像を整形 - ポーズプレビュー画像
            imageTrimmer = new ImageTrimmer();
            await imageTrimmer.loadByDataUrl(pose.poseImageDataUrl);
            await imageTrimmer.crop(0, trimmed.marginTop, trimmed.width, trimmed.heightNew);
            console.log(`[Pose] finalize - Trimmed margin of pose preview image...`, pose.timeMiliseconds, trimmed);
            await imageTrimmer.resizeWithFit({
                width: this.IMAGE_WIDTH,
            });
            newDataUrl = await imageTrimmer.getDataUrl('image/jpeg', this.IMAGE_JPEG_QUALITY);
            if (!newDataUrl) {
                console.warn(`[Pose] finalize - Could not get the new dataurl for pose preview image`);
                continue;
            }
            pose.poseImageDataUrl = newDataUrl;
        }
        this.isFinalized = true;
    }
    getSimilarPoses(results, threshold = 0.9) {
        const poseVector = Pose.getPoseVector(results.ea);
        if (!poseVector)
            throw 'Could not get the pose vector';
        const poses = [];
        for (const pose of this.poses) {
            const similarity = Pose.getPoseSimilarity(pose.vectors, poseVector);
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
        const similarity = Pose.getPoseSimilarity(poseVectorA, poseVectorB);
        if (similarity >= threshold)
            isSimilar = true;
        // console.log(`[Pose] isSimilarPose`, isSimilar, similarity);
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
        for (const pose of this.poses) {
            if (pose.frameImageDataUrl) {
                try {
                    const index = pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
                    const base64 = pose.frameImageDataUrl.substring(index);
                    jsZip.file(`frame-${pose.timeMiliseconds}.jpg`, base64, {
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
                    jsZip.file(`pose-${pose.timeMiliseconds}.jpg`, base64, {
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
                for (const key of Pose.POSE_VECTOR_MAPPINGS) {
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
        this.poses = parsedJson.poses.map((poseJsonItem) => {
            const poseVector = {};
            Pose.POSE_VECTOR_MAPPINGS.map((key, index) => {
                poseVector[key] = poseJsonItem.vectors[index];
            });
            return {
                timeMiliseconds: poseJsonItem.t,
                durationMiliseconds: poseJsonItem.d,
                pose: poseJsonItem.pose,
                vectors: poseVector,
                frameImageDataUrl: undefined,
            };
        });
    }
    async loadZip(buffer, includeImages = true) {
        console.log(`[Pose] loadZip...`, JSZip);
        const jsZip = new JSZip();
        console.log(`[Pose] init...`);
        const zip = await jsZip.loadAsync(buffer, { base64: false });
        if (!zip)
            throw 'ZIPファイルを読み込めませんでした';
        const json = await zip.file('poses.json')?.async('text');
        if (json === undefined) {
            throw 'ZIPファイルに pose.json が含まれていません';
        }
        this.loadJson(json);
        if (includeImages) {
            for (const pose of this.poses) {
                if (!pose.frameImageDataUrl) {
                    const frameImageFileName = `frame-${pose.timeMiliseconds}.jpg`;
                    const imageBase64 = await zip
                        .file(frameImageFileName)
                        ?.async('base64');
                    if (imageBase64) {
                        pose.frameImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
                    }
                }
                if (!pose.poseImageDataUrl) {
                    const poseImageFileName = `pose-${pose.timeMiliseconds}.jpg`;
                    const imageBase64 = await zip
                        .file(poseImageFileName)
                        ?.async('base64');
                    if (imageBase64) {
                        pose.poseImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
                    }
                }
            }
        }
    }
}
Pose.IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;
Pose.POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS5qcyIsInNvdXJjZVJvb3QiOiIiLCJzb3VyY2VzIjpbIi4uLy4uLy4uLy4uLy4uL3Byb2plY3RzL25neC1tcC1wb3NlLWV4dHJhY3Rvci9zcmMvbGliL2NsYXNzZXMvcG9zZS50cyJdLCJuYW1lcyI6W10sIm1hcHBpbmdzIjoiQUFBQSxPQUFPLEVBQUUsY0FBYyxFQUFXLE1BQU0scUJBQXFCLENBQUM7QUFDOUQsT0FBTyxLQUFLLEtBQUssTUFBTSxPQUFPLENBQUM7QUFNL0IsYUFBYTtBQUNiLE9BQU8sYUFBYSxNQUFNLGdCQUFnQixDQUFDO0FBRTNDLE9BQU8sRUFBRSxZQUFZLEVBQUUsTUFBTSwyQkFBMkIsQ0FBQztBQUV6RCxNQUFNLE9BQU8sSUFBSTtJQXdCZjtRQWZPLFVBQUssR0FBZSxFQUFFLENBQUM7UUFDdkIsZ0JBQVcsR0FBYSxLQUFLLENBQUM7UUFXcEIsdUJBQWtCLEdBQUcsR0FBRyxDQUFDO1FBQ3pCLGdCQUFXLEdBQUcsR0FBRyxDQUFDO1FBR2pDLElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7U0FDWixDQUFDO0lBQ0osQ0FBQztJQUVELFlBQVk7UUFDVixPQUFPLElBQUksQ0FBQyxhQUFhLENBQUMsSUFBSSxDQUFDO0lBQ2pDLENBQUM7SUFFRCxZQUFZLENBQUMsU0FBaUI7UUFDNUIsSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLEdBQUcsU0FBUyxDQUFDO0lBQ3RDLENBQUM7SUFFRCxnQkFBZ0IsQ0FBQyxLQUFhLEVBQUUsTUFBYyxFQUFFLFFBQWdCO1FBQzlELElBQUksQ0FBQyxhQUFhLENBQUMsS0FBSyxHQUFHLEtBQUssQ0FBQztRQUNqQyxJQUFJLENBQUMsYUFBYSxDQUFDLE1BQU0sR0FBRyxNQUFNLENBQUM7UUFDbkMsSUFBSSxDQUFDLGFBQWEsQ0FBQyxRQUFRLEdBQUcsUUFBUSxDQUFDO0lB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SUFBSSxDQUFDLGlCQUFpQixDQUFDO3dCQUN4QixFQUFFLEtBQUssQ0FBQyxRQUFRLENBQUMsQ0FBQztvQkFDcEIsSUFBSSxXQUFXLEVBQUU7d0JBQ2YsSUFBSSxDQUFDLGdCQUFnQixHQUFHLDBCQUEwQixXQUFXLEVBQUUsQ0FBQztxQkFDakU7aUJBQ0Y7YUFDRjtTQUNGO0lBQ0gsQ0FBQzs7QUFuZXNCLHdDQUFtQyxHQUFHLElBQUksQ0FBQztBQUUzQyx5QkFBb0IsR0FBRztJQUM1Qyx3QkFBd0I7SUFDeEIsMkJBQTJCO0lBQzNCLHNCQUFzQjtJQUN0Qix5QkFBeUI7Q0FDMUIsQ0FBQyIsInNvdXJjZXNDb250ZW50IjpbImltcG9ydCB7IFBPU0VfTEFORE1BUktTLCBSZXN1bHRzIH0gZnJvbSAnQG1lZGlhcGlwZS9ob2xpc3RpYyc7XG5pbXBvcnQgKiBhcyBKU1ppcCBmcm9tICdqc3ppcCc7XG5pbXBvcnQgeyBQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1pdGVtJztcbmltcG9ydCB7IFBvc2VKc29uIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLWpzb24nO1xuaW1wb3J0IHsgUG9zZUpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLWpzb24taXRlbSc7XG5pbXBvcnQgeyBQb3NlVmVjdG9yIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXZlY3Rvcic7XG5cbi8vIEB0cy1pZ25vcmVcbmltcG9ydCBjb3NTaW1pbGFyaXR5IGZyb20gJ2Nvcy1zaW1pbGFyaXR5JztcbmltcG9ydCB7IFNpbWlsYXJQb3NlSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvbWF0Y2hlZC1wb3NlLWl0ZW0nO1xuaW1wb3J0IHsgSW1hZ2VUcmltbWVyIH0gZnJvbSAnLi9pbnRlcm5hbHMvaW1hZ2UtdHJpbW1lcic7XG5cbmV4cG9ydCBjbGFzcyBQb3NlIHtcbiAgcHVibGljIGdlbmVyYXRvcj86IHN0cmluZztcbiAgcHVibGljIHZlcnNpb24/OiBudW1iZXI7XG4gIHByaXZhdGUgdmlkZW9NZXRhZGF0YSE6IHtcbiAgICBuYW1lOiBzdHJpbmc7XG4gICAgd2lkdGg6IG51bWJlcjtcbiAgICBoZWlnaHQ6IG51bWJlcjtcbiAgICBkdXJhdGlvbjogbnVtYmVyO1xuICB9O1xuICBwdWJsaWMgcG9zZXM6IFBvc2VJdGVtW10gPSBbXTtcbiAgcHVibGljIGlzRmluYWxpemVkPzogYm9vbGVhbiA9IGZhbHNlO1xuXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSVNfRU5BQkxFX0RVUExJQ0FURURfUE9TRV9SRURVQ1RJT04gPSB0cnVlO1xuXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgUE9TRV9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgJ3JpZ2h0V3Jpc3RUb1JpZ2h0RWxib3cnLFxuICAgICdyaWdodEVsYm93VG9SaWdodFNob3VsZGVyJyxcbiAgICAnbGVmdFdyaXN0VG9MZWZ0RWxib3cnLFxuICAgICdsZWZ0RWxib3dUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9KUEVHX1FVQUxJVFkgPSAwLjc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEggPSA5MDA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgZ2V0UG9zZXMoKTogUG9zZUl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgZ2V0UG9zZUJ5VGltZSh0aW1lTWlsaXNlY29uZHM6IG51bWJlcik6IFBvc2VJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UudGltZU1pbGlzZWNvbmRzID09PSB0aW1lTWlsaXNlY29uZHMpO1xuICB9XG5cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlSnBlZ0RhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBwb3NlSW1hZ2VKcGVnRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHZpZGVvV2lkdGg6IG51bWJlcixcbiAgICB2aWRlb0hlaWdodDogbnVtYmVyLFxuICAgIHZpZGVvRHVyYXRpb246IG51bWJlcixcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICkge1xuICAgIHRoaXMuc2V0VmlkZW9NZXRhRGF0YSh2aWRlb1dpZHRoLCB2aWRlb0hlaWdodCwgdmlkZW9EdXJhdGlvbik7XG5cbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHJldHVybjtcblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZV0gcHVzaFBvc2UgLSBDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHdpdGggdGhlIHdvcmxkIGNvb3JkaW5hdGVgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBQb3NlLmdldFBvc2VWZWN0b3IocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUpO1xuICAgIGlmICghcG9zZVZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VdIHB1c2hQb3NlIC0gQ291bGQgbm90IGdldCB0aGUgcG9zZSB2ZWN0b3JgLFxuICAgICAgICBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZVxuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBwb3NlOiBQb3NlSXRlbSA9IHtcbiAgICAgIHRpbWVNaWxpc2Vjb25kczogdmlkZW9UaW1lTWlsaXNlY29uZHMsXG4gICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiAtMSxcbiAgICAgIHBvc2U6IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLm1hcCgobGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtsYW5kbWFyay54LCBsYW5kbWFyay55LCBsYW5kbWFyay56LCBsYW5kbWFyay52aXNpYmlsaXR5XTtcbiAgICAgIH0pLFxuICAgICAgdmVjdG9yczogcG9zZVZlY3RvcixcbiAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBmcmFtZUltYWdlSnBlZ0RhdGFVcmwsXG4gICAgICBwb3NlSW1hZ2VEYXRhVXJsOiBwb3NlSW1hZ2VKcGVnRGF0YVVybCxcbiAgICB9O1xuXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgICAgaWYgKFBvc2UuaXNTaW1pbGFyUG9zZShsYXN0UG9zZS52ZWN0b3JzLCBwb3NlLnZlY3RvcnMpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgLy8g5YmN5Zue44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgICBjb25zdCBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHZpZGVvVGltZU1pbGlzZWNvbmRzIC0gbGFzdFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICB0aGlzLnBvc2VzLnB1c2gocG9zZSk7XG4gIH1cblxuICBhc3luYyBmaW5hbGl6ZSgpIHtcbiAgICBpZiAoMCA9PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g5YWo44Od44O844K644KS5q+U6LyD44GX44Gm6aGe5Ly844Od44O844K644KS5YmK6ZmkXG4gICAgaWYgKFBvc2UuSVNfRU5BQkxFX0RVUExJQ0FURURfUE9TRV9SRURVQ1RJT04pIHtcbiAgICAgIGNvbnN0IG5ld1Bvc2VzOiBQb3NlSXRlbVtdID0gW107XG4gICAgICBmb3IgKGNvbnN0IHBvc2VBIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgICAgbGV0IGlzRHVwbGljYXRlZCA9IGZhbHNlO1xuICAgICAgICBmb3IgKGNvbnN0IHBvc2VCIG9mIG5ld1Bvc2VzKSB7XG4gICAgICAgICAgaWYgKFBvc2UuaXNTaW1pbGFyUG9zZShwb3NlQS52ZWN0b3JzLCBwb3NlQi52ZWN0b3JzKSkge1xuICAgICAgICAgICAgaXNEdXBsaWNhdGVkID0gdHJ1ZTtcbiAgICAgICAgICAgIGJyZWFrO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoaXNEdXBsaWNhdGVkKSBjb250aW51ZTtcblxuICAgICAgICBuZXdQb3Nlcy5wdXNoKHBvc2VBKTtcbiAgICAgIH1cblxuICAgICAgY29uc29sZS5pbmZvKFxuICAgICAgICBgW1Bvc2VdIGdldEpzb24gLSBSZWR1Y2VkICR7dGhpcy5wb3Nlcy5sZW5ndGh9IHBvc2VzIC0+ICR7bmV3UG9zZXMubGVuZ3RofSBwb3Nlc2BcbiAgICAgICk7XG4gICAgICB0aGlzLnBvc2VzID0gbmV3UG9zZXM7XG4gICAgfVxuXG4gICAgLy8g5pyA5b6M44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgICAgaWYgKGxhc3RQb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMgPT0gLTEpIHtcbiAgICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOeUu+WDj+OCkuaVtOW9olxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsIHx8ICFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODleODrOODvOODoOeUu+WDj1xuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZV0gZmluYWxpemUgLSBQcm9jZXNzaW5nIGZyYW1lIGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHNcbiAgICAgICk7XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgY29uc3QgbWFyZ2luQ29sb3IgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0TWFyZ2luQ29sb3IoKTtcbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VdIGZpbmFsaXplIC0gRGV0ZWN0ZWQgbWFyZ2luIGNvbG9yLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgIG1hcmdpbkNvbG9yXG4gICAgICApO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yID09PSBudWxsKSBjb250aW51ZTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciAhPT0gJyMwMDAwMDAnKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VdIGZpbmFsaXplIC0gU2tpcCB0aGlzIGZyYW1lIGltYWdlLCBiZWNhdXNlIHRoZSBtYXJnaW4gY29sb3IgaXMgbm90IGJsYWNrLmBcbiAgICAgICAgKTtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIGNvbnN0IHRyaW1tZWQgPSBhd2FpdCBpbWFnZVRyaW1tZXIudHJpbU1hcmdpbihtYXJnaW5Db2xvcik7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlXSBmaW5hbGl6ZSAtIFRyaW1tZWQgbWFyZ2luIG9mIGZyYW1lIGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgIHRyaW1tZWRcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbGV0IG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgJ2ltYWdlL2pwZWcnLFxuICAgICAgICB0aGlzLklNQUdFX0pQRUdfUVVBTElUWVxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlXSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBmcmFtZSBpbWFnZWBcbiAgICAgICAgKTtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcblxuICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g44Od44O844K644OX44Os44OT44Ol44O855S75YOPXG4gICAgICBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLnBvc2VJbWFnZURhdGFVcmwpO1xuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgMCxcbiAgICAgICAgdHJpbW1lZC5tYXJnaW5Ub3AsXG4gICAgICAgIHRyaW1tZWQud2lkdGgsXG4gICAgICAgIHRyaW1tZWQuaGVpZ2h0TmV3XG4gICAgICApO1xuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZV0gZmluYWxpemUgLSBUcmltbWVkIG1hcmdpbiBvZiBwb3NlIHByZXZpZXcgaW1hZ2UuLi5gLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgdHJpbW1lZFxuICAgICAgKTtcblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlc2l6ZVdpdGhGaXQoe1xuICAgICAgICB3aWR0aDogdGhpcy5JTUFHRV9XSURUSCxcbiAgICAgIH0pO1xuXG4gICAgICBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgICdpbWFnZS9qcGVnJyxcbiAgICAgICAgdGhpcy5JTUFHRV9KUEVHX1FVQUxJVFlcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZV0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgcG9zZSBwcmV2aWV3IGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG4gICAgfVxuXG4gICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gIH1cblxuICBnZXRTaW1pbGFyUG9zZXMoXG4gICAgcmVzdWx0czogUmVzdWx0cyxcbiAgICB0aHJlc2hvbGQ6IG51bWJlciA9IDAuOVxuICApOiBTaW1pbGFyUG9zZUl0ZW1bXSB7XG4gICAgY29uc3QgcG9zZVZlY3RvciA9IFBvc2UuZ2V0UG9zZVZlY3RvcigocmVzdWx0cyBhcyBhbnkpLmVhKTtcbiAgICBpZiAoIXBvc2VWZWN0b3IpIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHZlY3Rvcic7XG5cbiAgICBjb25zdCBwb3NlcyA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZS5nZXRQb3NlU2ltaWxhcml0eShwb3NlLnZlY3RvcnMsIHBvc2VWZWN0b3IpO1xuICAgICAgaWYgKHRocmVzaG9sZCA8PSBzaW1pbGFyaXR5KSB7XG4gICAgICAgIHBvc2VzLnB1c2goe1xuICAgICAgICAgIC4uLnBvc2UsXG4gICAgICAgICAgc2ltaWxhcml0eTogc2ltaWxhcml0eSxcbiAgICAgICAgfSk7XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIHBvc2VzO1xuICB9XG5cbiAgc3RhdGljIGdldFBvc2VWZWN0b3IoXG4gICAgcG9zZUxhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXVxuICApOiBQb3NlVmVjdG9yIHwgdW5kZWZpbmVkIHtcbiAgICByZXR1cm4ge1xuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56LFxuICAgICAgXSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgIH07XG4gIH1cblxuICBzdGF0aWMgaXNTaW1pbGFyUG9zZShcbiAgICBwb3NlVmVjdG9yQTogUG9zZVZlY3RvcixcbiAgICBwb3NlVmVjdG9yQjogUG9zZVZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjlcbiAgKTogYm9vbGVhbiB7XG4gICAgbGV0IGlzU2ltaWxhciA9IGZhbHNlO1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlLmdldFBvc2VTaW1pbGFyaXR5KHBvc2VWZWN0b3JBLCBwb3NlVmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkKSBpc1NpbWlsYXIgPSB0cnVlO1xuXG4gICAgLy8gY29uc29sZS5sb2coYFtQb3NlXSBpc1NpbWlsYXJQb3NlYCwgaXNTaW1pbGFyLCBzaW1pbGFyaXR5KTtcblxuICAgIHJldHVybiBpc1NpbWlsYXI7XG4gIH1cblxuICBzdGF0aWMgZ2V0UG9zZVNpbWlsYXJpdHkoXG4gICAgcG9zZVZlY3RvckE6IFBvc2VWZWN0b3IsXG4gICAgcG9zZVZlY3RvckI6IFBvc2VWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXMgPSB7XG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgcG9zZVZlY3RvckEubGVmdFdyaXN0VG9MZWZ0RWxib3csXG4gICAgICAgIHBvc2VWZWN0b3JCLmxlZnRXcmlzdFRvTGVmdEVsYm93XG4gICAgICApLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIHBvc2VWZWN0b3JBLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyLFxuICAgICAgICBwb3NlVmVjdG9yQi5sZWZ0RWxib3dUb0xlZnRTaG91bGRlclxuICAgICAgKSxcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIHBvc2VWZWN0b3JBLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3csXG4gICAgICAgIHBvc2VWZWN0b3JCLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3dcbiAgICAgICksXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBwb3NlVmVjdG9yQS5yaWdodEVsYm93VG9SaWdodFNob3VsZGVyLFxuICAgICAgICBwb3NlVmVjdG9yQi5yaWdodEVsYm93VG9SaWdodFNob3VsZGVyXG4gICAgICApLFxuICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNTdW0gPSBPYmplY3QudmFsdWVzKGNvc1NpbWlsYXJpdGllcykucmVkdWNlKFxuICAgICAgKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLFxuICAgICAgMFxuICAgICk7XG4gICAgcmV0dXJuIGNvc1NpbWlsYXJpdGllc1N1bSAvIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllcykubGVuZ3RoO1xuICB9XG5cbiAgcHVibGljIGFzeW5jIGdldFppcCgpOiBQcm9taXNlPEJsb2I+IHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGpzWmlwLmZpbGUoJ3Bvc2VzLmpzb24nLCBhd2FpdCB0aGlzLmdldEpzb24oKSk7XG5cbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LmpwZ2AsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UucG9zZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uanBnYCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICBwdWJsaWMgYXN5bmMgZ2V0SnNvbigpOiBQcm9taXNlPHN0cmluZz4ge1xuICAgIGlmICh0aGlzLnZpZGVvTWV0YWRhdGEgPT09IHVuZGVmaW5lZCB8fCB0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpXG4gICAgICByZXR1cm4gJ3t9JztcblxuICAgIGlmICghdGhpcy5pc0ZpbmFsaXplZCkge1xuICAgICAgYXdhaXQgdGhpcy5maW5hbGl6ZSgpO1xuICAgIH1cblxuICAgIGxldCBwb3NlTGFuZG1hcmtNYXBwaW5ncyA9IFtdO1xuICAgIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKFBPU0VfTEFORE1BUktTKSkge1xuICAgICAgY29uc3QgaW5kZXg6IG51bWJlciA9IFBPU0VfTEFORE1BUktTW2tleSBhcyBrZXlvZiB0eXBlb2YgUE9TRV9MQU5ETUFSS1NdO1xuICAgICAgcG9zZUxhbmRtYXJrTWFwcGluZ3NbaW5kZXhdID0ga2V5O1xuICAgIH1cblxuICAgIGNvbnN0IGpzb246IFBvc2VKc29uID0ge1xuICAgICAgZ2VuZXJhdG9yOiAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InLFxuICAgICAgdmVyc2lvbjogMSxcbiAgICAgIHZpZGVvOiB0aGlzLnZpZGVvTWV0YWRhdGEhLFxuICAgICAgcG9zZXM6IHRoaXMucG9zZXMubWFwKChwb3NlOiBQb3NlSXRlbSk6IFBvc2VKc29uSXRlbSA9PiB7XG4gICAgICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBbXTtcbiAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZS5QT1NFX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgIHBvc2VWZWN0b3IucHVzaChwb3NlLnZlY3RvcnNba2V5IGFzIGtleW9mIFBvc2VWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdDogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgZDogcG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIHBvc2U6IHBvc2UucG9zZSxcbiAgICAgICAgICB2ZWN0b3JzOiBwb3NlVmVjdG9yLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcChcbiAgICAgIChwb3NlSnNvbkl0ZW06IFBvc2VKc29uSXRlbSk6IFBvc2VJdGVtID0+IHtcbiAgICAgICAgY29uc3QgcG9zZVZlY3RvcjogYW55ID0ge307XG4gICAgICAgIFBvc2UuUE9TRV9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgICAgcG9zZVZlY3RvcltrZXkgYXMga2V5b2YgUG9zZVZlY3Rvcl0gPSBwb3NlSnNvbkl0ZW0udmVjdG9yc1tpbmRleF07XG4gICAgICAgIH0pO1xuXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdGltZU1pbGlzZWNvbmRzOiBwb3NlSnNvbkl0ZW0udCxcbiAgICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBwb3NlSnNvbkl0ZW0uZCxcbiAgICAgICAgICBwb3NlOiBwb3NlSnNvbkl0ZW0ucG9zZSxcbiAgICAgICAgICB2ZWN0b3JzOiBwb3NlVmVjdG9yLFxuICAgICAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiB1bmRlZmluZWQsXG4gICAgICAgIH07XG4gICAgICB9XG4gICAgKTtcbiAgfVxuXG4gIGFzeW5jIGxvYWRaaXAoYnVmZmVyOiBBcnJheUJ1ZmZlciwgaW5jbHVkZUltYWdlczogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VdIGxvYWRaaXAuLi5gLCBKU1ppcCk7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VdIGluaXQuLi5gKTtcbiAgICBjb25zdCB6aXAgPSBhd2FpdCBqc1ppcC5sb2FkQXN5bmMoYnVmZmVyLCB7IGJhc2U2NDogZmFsc2UgfSk7XG4gICAgaWYgKCF6aXApIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgpLoqq3jgb/ovrzjgoHjgb7jgZvjgpPjgafjgZfjgZ8nO1xuXG4gICAgY29uc3QganNvbiA9IGF3YWl0IHppcC5maWxlKCdwb3Nlcy5qc29uJyk/LmFzeW5jKCd0ZXh0Jyk7XG4gICAgaWYgKGpzb24gPT09IHVuZGVmaW5lZCkge1xuICAgICAgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OBqyBwb3NlLmpzb24g44GM5ZCr44G+44KM44Gm44GE44G+44Gb44KTJztcbiAgICB9XG5cbiAgICB0aGlzLmxvYWRKc29uKGpzb24pO1xuXG4gICAgaWYgKGluY2x1ZGVJbWFnZXMpIHtcbiAgICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IGZyYW1lSW1hZ2VGaWxlTmFtZSA9IGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS5qcGdgO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShmcmFtZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBgZGF0YTppbWFnZS9qcGVnO2Jhc2U2NCwke2ltYWdlQmFzZTY0fWA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGlmICghcG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc3QgcG9zZUltYWdlRmlsZU5hbWUgPSBgcG9zZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS5qcGdgO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShwb3NlSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gYGRhdGE6aW1hZ2UvanBlZztiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxufVxuIl19