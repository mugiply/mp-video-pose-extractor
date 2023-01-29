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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBRXpELE1BQU0sT0FBTyxPQUFPO0lBbUNsQjtRQXpCTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQVNyQyxhQUFhO1FBQ0ksZ0JBQVcsR0FBVyxJQUFJLENBQUM7UUFDM0IsZUFBVSxHQUN6QixZQUFZLENBQUM7UUFDRSxrQkFBYSxHQUFHLEdBQUcsQ0FBQztRQUVyQyxVQUFVO1FBQ08sZ0NBQTJCLEdBQUcsU0FBUyxDQUFDO1FBQ3hDLHlDQUFvQyxHQUFHLEVBQUUsQ0FBQztRQUUzRCxXQUFXO1FBQ00sdUNBQWtDLEdBQUcsU0FBUyxDQUFDO1FBQy9DLHVDQUFrQyxHQUFHLFdBQVcsQ0FBQztRQUNqRCw0Q0FBdUMsR0FBRyxHQUFHLENBQUM7UUFHN0QsSUFBSSxDQUFDLGFBQWEsR0FBRztZQUNuQixJQUFJLEVBQUUsRUFBRTtZQUNSLEtBQUssRUFBRSxDQUFDO1lBQ1IsTUFBTSxFQUFFLENBQUM7WUFDVCxRQUFRLEVBQUUsQ0FBQztZQUNYLHFCQUFxQixFQUFFLENBQUM7U0FDekIsQ0FBQztJQUNKLENBQUM7SUFFRCxZQUFZO1FBQ1YsT0FBTyxJQUFJLENBQUMsYUFBYSxDQUFDLElBQUksQ0FBQztJQUNqQyxDQUFDO0lBRUQsWUFBWSxDQUFDLFNBQWlCO1FBQzVCLElBQUks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VSxXQUFXLFdBQVcsRUFBRSxDQUFDO3FCQUMxRTtpQkFDRjtnQkFDRCxJQUFJLENBQUMsSUFBSSxDQUFDLGdCQUFnQixFQUFFO29CQUMxQixNQUFNLGlCQUFpQixHQUFHLFFBQVEsSUFBSSxDQUFDLGVBQWUsSUFBSSxPQUFPLEVBQUUsQ0FBQztvQkFDcEUsTUFBTSxXQUFXLEdBQUcsTUFBTSxHQUFHO3lCQUMxQixJQUFJLENBQUMsaUJBQWlCLENBQUM7d0JBQ3hCLEVBQUUsS0FBSyxDQUFDLFFBQVEsQ0FBQyxDQUFDO29CQUNwQixJQUFJLFdBQVcsRUFBRTt3QkFDZixJQUFJLENBQUMsZ0JBQWdCLEdBQUcsUUFBUSxJQUFJLENBQUMsVUFBVSxXQUFXLFdBQVcsRUFBRSxDQUFDO3FCQUN6RTtpQkFDRjthQUNGO1NBQ0Y7SUFDSCxDQUFDOztBQWhrQnNCLDRCQUFvQixHQUFHO0lBQzVDLHdCQUF3QjtJQUN4QiwyQkFBMkI7SUFDM0Isc0JBQXNCO0lBQ3RCLHlCQUF5QjtDQUMxQixDQUFDIiwic291cmNlc0NvbnRlbnQiOlsiaW1wb3J0IHsgUE9TRV9MQU5ETUFSS1MsIFJlc3VsdHMgfSBmcm9tICdAbWVkaWFwaXBlL2hvbGlzdGljJztcbmltcG9ydCAqIGFzIEpTWmlwIGZyb20gJ2pzemlwJztcbmltcG9ydCB7IFBvc2VTZXRJdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1pdGVtJztcbmltcG9ydCB7IFBvc2VTZXRKc29uIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uJztcbmltcG9ydCB7IFBvc2VTZXRKc29uSXRlbSB9IGZyb20gJy4uL2ludGVyZmFjZXMvcG9zZS1zZXQtanNvbi1pdGVtJztcbmltcG9ydCB7IFBvc2VWZWN0b3IgfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2UtdmVjdG9yJztcblxuLy8gQHRzLWlnbm9yZVxuaW1wb3J0IGNvc1NpbWlsYXJpdHkgZnJvbSAnY29zLXNpbWlsYXJpdHknO1xuaW1wb3J0IHsgU2ltaWxhclBvc2VJdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9tYXRjaGVkLXBvc2UtaXRlbSc7XG5pbXBvcnQgeyBJbWFnZVRyaW1tZXIgfSBmcm9tICcuL2ludGVybmFscy9pbWFnZS10cmltbWVyJztcblxuZXhwb3J0IGNsYXNzIFBvc2VTZXQge1xuICBwdWJsaWMgZ2VuZXJhdG9yPzogc3RyaW5nO1xuICBwdWJsaWMgdmVyc2lvbj86IG51bWJlcjtcbiAgcHJpdmF0ZSB2aWRlb01ldGFkYXRhIToge1xuICAgIG5hbWU6IHN0cmluZztcbiAgICB3aWR0aDogbnVtYmVyO1xuICAgIGhlaWdodDogbnVtYmVyO1xuICAgIGR1cmF0aW9uOiBudW1iZXI7XG4gICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiBudW1iZXI7XG4gIH07XG4gIHB1YmxpYyBwb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICBwdWJsaWMgaXNGaW5hbGl6ZWQ/OiBib29sZWFuID0gZmFsc2U7XG5cbiAgcHVibGljIHN0YXRpYyByZWFkb25seSBQT1NFX1ZFQ1RPUl9NQVBQSU5HUyA9IFtcbiAgICAncmlnaHRXcmlzdFRvUmlnaHRFbGJvdycsXG4gICAgJ3JpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXInLFxuICAgICdsZWZ0V3Jpc3RUb0xlZnRFbGJvdycsXG4gICAgJ2xlZnRFbGJvd1RvTGVmdFNob3VsZGVyJyxcbiAgXTtcblxuICAvLyDnlLvlg4/mm7jjgY3lh7rjgZfmmYLjga7oqK3lrppcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9XSURUSDogbnVtYmVyID0gMTA4MDtcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NSU1FOiAnaW1hZ2UvanBlZycgfCAnaW1hZ2UvcG5nJyB8ICdpbWFnZS93ZWJwJyA9XG4gICAgJ2ltYWdlL3dlYnAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX1FVQUxJVFkgPSAwLjg7XG5cbiAgLy8g55S75YOP44Gu5L2Z55m96Zmk5Y67XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SID0gJyMwMDAwMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRCA9IDUwO1xuXG4gIC8vIOeUu+WDj+OBruiDjOaZr+iJsue9ruaPm1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IgPSAnIzAxNkFGRCc7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUiA9ICcjRkZGRkZGMDAnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRCA9IDEzMDtcblxuICBjb25zdHJ1Y3RvcigpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSB7XG4gICAgICBuYW1lOiAnJyxcbiAgICAgIHdpZHRoOiAwLFxuICAgICAgaGVpZ2h0OiAwLFxuICAgICAgZHVyYXRpb246IDAsXG4gICAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IDAsXG4gICAgfTtcbiAgfVxuXG4gIGdldFZpZGVvTmFtZSgpIHtcbiAgICByZXR1cm4gdGhpcy52aWRlb01ldGFkYXRhLm5hbWU7XG4gIH1cblxuICBzZXRWaWRlb05hbWUodmlkZW9OYW1lOiBzdHJpbmcpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEubmFtZSA9IHZpZGVvTmFtZTtcbiAgfVxuXG4gIHNldFZpZGVvTWV0YURhdGEod2lkdGg6IG51bWJlciwgaGVpZ2h0OiBudW1iZXIsIGR1cmF0aW9uOiBudW1iZXIpIHtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEud2lkdGggPSB3aWR0aDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuaGVpZ2h0ID0gaGVpZ2h0O1xuICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiA9IGR1cmF0aW9uO1xuICB9XG5cbiAgZ2V0TnVtYmVyT2ZQb3NlcygpOiBudW1iZXIge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiAtMTtcbiAgICByZXR1cm4gdGhpcy5wb3Nlcy5sZW5ndGg7XG4gIH1cblxuICBnZXRQb3NlcygpOiBQb3NlU2V0SXRlbVtdIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gW107XG4gICAgcmV0dXJuIHRoaXMucG9zZXM7XG4gIH1cblxuICBnZXRQb3NlQnlUaW1lKHRpbWVNaWxpc2Vjb25kczogbnVtYmVyKTogUG9zZVNldEl0ZW0gfCB1bmRlZmluZWQge1xuICAgIGlmICh0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpIHJldHVybiB1bmRlZmluZWQ7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMuZmluZCgocG9zZSkgPT4gcG9zZS50aW1lTWlsaXNlY29uZHMgPT09IHRpbWVNaWxpc2Vjb25kcyk7XG4gIH1cblxuICBwdXNoUG9zZShcbiAgICB2aWRlb1RpbWVNaWxpc2Vjb25kczogbnVtYmVyLFxuICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiBzdHJpbmcgfCB1bmRlZmluZWQsXG4gICAgcG9zZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHZpZGVvV2lkdGg6IG51bWJlcixcbiAgICB2aWRlb0hlaWdodDogbnVtYmVyLFxuICAgIHZpZGVvRHVyYXRpb246IG51bWJlcixcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICkge1xuICAgIHRoaXMuc2V0VmlkZW9NZXRhRGF0YSh2aWRlb1dpZHRoLCB2aWRlb0hlaWdodCwgdmlkZW9EdXJhdGlvbik7XG5cbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHJldHVybjtcblxuICAgIGlmICh0aGlzLnBvc2VzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhpcy52aWRlb01ldGFkYXRhLmZpcnN0UG9zZURldGVjdGVkVGltZSA9IHZpZGVvVGltZU1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgLSBDb3VsZCBub3QgZ2V0IHRoZSBwb3NlIHdpdGggdGhlIHdvcmxkIGNvb3JkaW5hdGVgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBQb3NlU2V0LmdldFBvc2VWZWN0b3IocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUpO1xuICAgIGlmICghcG9zZVZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlIC0gQ291bGQgbm90IGdldCB0aGUgcG9zZSB2ZWN0b3JgLFxuICAgICAgICBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZVxuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBwb3NlOiBQb3NlU2V0SXRlbSA9IHtcbiAgICAgIHRpbWVNaWxpc2Vjb25kczogdmlkZW9UaW1lTWlsaXNlY29uZHMsXG4gICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiAtMSxcbiAgICAgIHBvc2U6IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLm1hcCgod29ybGRDb29yZGluYXRlTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay54LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnksXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueixcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay52aXNpYmlsaXR5LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICBsZWZ0SGFuZDogcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcz8ubWFwKChub3JtYWxpemVkTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueCxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueSxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueixcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgcmlnaHRIYW5kOiByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICB2ZWN0b3JzOiBwb3NlVmVjdG9yLFxuICAgICAgZnJhbWVJbWFnZURhdGFVcmw6IGZyYW1lSW1hZ2VEYXRhVXJsLFxuICAgICAgcG9zZUltYWdlRGF0YVVybDogcG9zZUltYWdlRGF0YVVybCxcbiAgICB9O1xuXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgICAgaWYgKFBvc2VTZXQuaXNTaW1pbGFyUG9zZShsYXN0UG9zZS52ZWN0b3JzLCBwb3NlLnZlY3RvcnMpKSB7XG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgLy8g5YmN5Zue44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgICBjb25zdCBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHZpZGVvVGltZU1pbGlzZWNvbmRzIC0gbGFzdFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICB0aGlzLnBvc2VzLnB1c2gocG9zZSk7XG4gIH1cblxuICBhc3luYyBmaW5hbGl6ZSgpIHtcbiAgICBpZiAoMCA9PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g5pyA5b6M44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgICAgaWYgKGxhc3RQb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMgPT0gLTEpIHtcbiAgICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOmHjeikh+ODneODvOOCuuOCkumZpOWOu1xuICAgIHRoaXMucmVtb3ZlRHVwbGljYXRlZFBvc2VzKCk7XG5cbiAgICAvLyDnlLvlg4/jga7jg57jg7zjgrjjg7PjgpLlj5blvpdcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0aW5nIGltYWdlIG1hcmdpbnMuLi5gKTtcbiAgICBsZXQgaW1hZ2VUcmltbWluZzpcbiAgICAgIHwge1xuICAgICAgICAgIG1hcmdpblRvcDogbnVtYmVyO1xuICAgICAgICAgIG1hcmdpbkJvdHRvbTogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE5ldzogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE9sZDogbnVtYmVyO1xuICAgICAgICAgIHdpZHRoOiBudW1iZXI7XG4gICAgICAgIH1cbiAgICAgIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGNvbnN0IG1hcmdpbkNvbG9yID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldE1hcmdpbkNvbG9yKCk7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGVkIG1hcmdpbiBjb2xvci4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICBtYXJnaW5Db2xvclxuICAgICAgKTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciA9PT0gbnVsbCkgY29udGludWU7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgIT09IHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgY29uc3QgdHJpbW1lZCA9IGF3YWl0IGltYWdlVHJpbW1lci50cmltTWFyZ2luKFxuICAgICAgICBtYXJnaW5Db2xvcixcbiAgICAgICAgdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG4gICAgICBpZiAoIXRyaW1tZWQpIGNvbnRpbnVlO1xuICAgICAgaW1hZ2VUcmltbWluZyA9IHRyaW1tZWQ7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVybWluZWQgaW1hZ2UgdHJpbW1pbmcgcG9zaXRpb25zLi4uYCxcbiAgICAgICAgdHJpbW1lZFxuICAgICAgKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIC8vIOeUu+WDj+OCkuaVtOW9olxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsIHx8ICFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gUHJvY2Vzc2luZyBpbWFnZS4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICBpZiAoaW1hZ2VUcmltbWluZykge1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgICAwLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcubWFyZ2luVG9wLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcud2lkdGgsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5oZWlnaHROZXdcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlcGxhY2VDb2xvcihcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SLFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIGxldCBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICApO1xuICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIGZyYW1lIGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg53jg7zjgrrjg5fjg6zjg5Pjg6Xjg7znlLvlg49cbiAgICAgIGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UucG9zZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgcG9zZSBwcmV2aWV3IGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG4gICAgfVxuXG4gICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gIH1cblxuICByZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTogdm9pZCB7XG4gICAgLy8g5YWo44Od44O844K644KS5q+U6LyD44GX44Gm6aGe5Ly844Od44O844K644KS5YmK6ZmkXG4gICAgY29uc3QgbmV3UG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IHBvc2VBIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpc0R1cGxpY2F0ZWQgPSBmYWxzZTtcbiAgICAgIGZvciAoY29uc3QgcG9zZUIgb2YgbmV3UG9zZXMpIHtcbiAgICAgICAgaWYgKFBvc2VTZXQuaXNTaW1pbGFyUG9zZShwb3NlQS52ZWN0b3JzLCBwb3NlQi52ZWN0b3JzKSkge1xuICAgICAgICAgIGlzRHVwbGljYXRlZCA9IHRydWU7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChpc0R1cGxpY2F0ZWQpIGNvbnRpbnVlO1xuXG4gICAgICBuZXdQb3Nlcy5wdXNoKHBvc2VBKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmluZm8oXG4gICAgICBgW1Bvc2VTZXRdIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcyAtIFJlZHVjZWQgJHt0aGlzLnBvc2VzLmxlbmd0aH0gcG9zZXMgLT4gJHtuZXdQb3Nlcy5sZW5ndGh9IHBvc2VzYFxuICAgICk7XG4gICAgdGhpcy5wb3NlcyA9IG5ld1Bvc2VzO1xuICB9XG5cbiAgZ2V0U2ltaWxhclBvc2VzKFxuICAgIHJlc3VsdHM6IFJlc3VsdHMsXG4gICAgdGhyZXNob2xkOiBudW1iZXIgPSAwLjlcbiAgKTogU2ltaWxhclBvc2VJdGVtW10ge1xuICAgIGNvbnN0IHBvc2VWZWN0b3IgPSBQb3NlU2V0LmdldFBvc2VWZWN0b3IoKHJlc3VsdHMgYXMgYW55KS5lYSk7XG4gICAgaWYgKCFwb3NlVmVjdG9yKSB0aHJvdyAnQ291bGQgbm90IGdldCB0aGUgcG9zZSB2ZWN0b3InO1xuXG4gICAgY29uc3QgcG9zZXMgPSBbXTtcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0UG9zZVNpbWlsYXJpdHkocG9zZS52ZWN0b3JzLCBwb3NlVmVjdG9yKTtcbiAgICAgIGlmICh0aHJlc2hvbGQgPD0gc2ltaWxhcml0eSkge1xuICAgICAgICBwb3Nlcy5wdXNoKHtcbiAgICAgICAgICAuLi5wb3NlLFxuICAgICAgICAgIHNpbWlsYXJpdHk6IHNpbWlsYXJpdHksXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBwb3NlcztcbiAgfVxuXG4gIHN0YXRpYyBnZXRQb3NlVmVjdG9yKFxuICAgIHBvc2VMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogUG9zZVZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgc3RhdGljIGlzU2ltaWxhclBvc2UoXG4gICAgcG9zZVZlY3RvckE6IFBvc2VWZWN0b3IsXG4gICAgcG9zZVZlY3RvckI6IFBvc2VWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC45XG4gICk6IGJvb2xlYW4ge1xuICAgIGxldCBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRQb3NlU2ltaWxhcml0eShwb3NlVmVjdG9yQSwgcG9zZVZlY3RvckIpO1xuICAgIGlmIChzaW1pbGFyaXR5ID49IHRocmVzaG9sZCkgaXNTaW1pbGFyID0gdHJ1ZTtcblxuICAgIC8vIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gaXNTaW1pbGFyUG9zZWAsIGlzU2ltaWxhciwgc2ltaWxhcml0eSk7XG5cbiAgICByZXR1cm4gaXNTaW1pbGFyO1xuICB9XG5cbiAgc3RhdGljIGdldFBvc2VTaW1pbGFyaXR5KFxuICAgIHBvc2VWZWN0b3JBOiBQb3NlVmVjdG9yLFxuICAgIHBvc2VWZWN0b3JCOiBQb3NlVmVjdG9yXG4gICk6IG51bWJlciB7XG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzID0ge1xuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIHBvc2VWZWN0b3JBLmxlZnRXcmlzdFRvTGVmdEVsYm93LFxuICAgICAgICBwb3NlVmVjdG9yQi5sZWZ0V3Jpc3RUb0xlZnRFbGJvd1xuICAgICAgKSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBwb3NlVmVjdG9yQS5sZWZ0RWxib3dUb0xlZnRTaG91bGRlcixcbiAgICAgICAgcG9zZVZlY3RvckIubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXJcbiAgICAgICksXG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBwb3NlVmVjdG9yQS5yaWdodFdyaXN0VG9SaWdodEVsYm93LFxuICAgICAgICBwb3NlVmVjdG9yQi5yaWdodFdyaXN0VG9SaWdodEVsYm93XG4gICAgICApLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgcG9zZVZlY3RvckEucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcixcbiAgICAgICAgcG9zZVZlY3RvckIucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlclxuICAgICAgKSxcbiAgICB9O1xuXG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzU3VtID0gT2JqZWN0LnZhbHVlcyhjb3NTaW1pbGFyaXRpZXMpLnJlZHVjZShcbiAgICAgIChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSxcbiAgICAgIDBcbiAgICApO1xuICAgIHJldHVybiBjb3NTaW1pbGFyaXRpZXNTdW0gLyBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXMpLmxlbmd0aDtcbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRaaXAoKTogUHJvbWlzZTxCbG9iPiB7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBqc1ppcC5maWxlKCdwb3Nlcy5qc29uJywgYXdhaXQgdGhpcy5nZXRKc29uKCkpO1xuXG4gICAgY29uc3QgaW1hZ2VGaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UucG9zZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICBnZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKElNQUdFX01JTUU6IHN0cmluZykge1xuICAgIHN3aXRjaCAoSU1BR0VfTUlNRSkge1xuICAgICAgY2FzZSAnaW1hZ2UvcG5nJzpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgICAgY2FzZSAnaW1hZ2UvanBlZyc6XG4gICAgICAgIHJldHVybiAnanBnJztcbiAgICAgIGNhc2UgJ2ltYWdlL3dlYnAnOlxuICAgICAgICByZXR1cm4gJ3dlYnAnO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgIH1cbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRKc29uKCk6IFByb21pc2U8c3RyaW5nPiB7XG4gICAgaWYgKHRoaXMudmlkZW9NZXRhZGF0YSA9PT0gdW5kZWZpbmVkIHx8IHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZClcbiAgICAgIHJldHVybiAne30nO1xuXG4gICAgaWYgKCF0aGlzLmlzRmluYWxpemVkKSB7XG4gICAgICBhd2FpdCB0aGlzLmZpbmFsaXplKCk7XG4gICAgfVxuXG4gICAgbGV0IHBvc2VMYW5kbWFya01hcHBpbmdzID0gW107XG4gICAgZm9yIChjb25zdCBrZXkgb2YgT2JqZWN0LmtleXMoUE9TRV9MQU5ETUFSS1MpKSB7XG4gICAgICBjb25zdCBpbmRleDogbnVtYmVyID0gUE9TRV9MQU5ETUFSS1Nba2V5IGFzIGtleW9mIHR5cGVvZiBQT1NFX0xBTkRNQVJLU107XG4gICAgICBwb3NlTGFuZG1hcmtNYXBwaW5nc1tpbmRleF0gPSBrZXk7XG4gICAgfVxuXG4gICAgY29uc3QganNvbjogUG9zZVNldEpzb24gPSB7XG4gICAgICBnZW5lcmF0b3I6ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicsXG4gICAgICB2ZXJzaW9uOiAxLFxuICAgICAgdmlkZW86IHRoaXMudmlkZW9NZXRhZGF0YSEsXG4gICAgICBwb3NlczogdGhpcy5wb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKTogUG9zZVNldEpzb25JdGVtID0+IHtcbiAgICAgICAgY29uc3QgcG9zZVZlY3RvciA9IFtdO1xuICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LlBPU0VfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgcG9zZVZlY3Rvci5wdXNoKHBvc2UudmVjdG9yc1trZXkgYXMga2V5b2YgUG9zZVZlY3Rvcl0pO1xuICAgICAgICB9XG5cbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICB0OiBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgICBkOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgcDogcG9zZS5wb3NlLFxuICAgICAgICAgIGw6IHBvc2UubGVmdEhhbmQsXG4gICAgICAgICAgcjogcG9zZS5yaWdodEhhbmQsXG4gICAgICAgICAgdjogcG9zZVZlY3RvcixcbiAgICAgICAgfTtcbiAgICAgIH0pLFxuICAgICAgcG9zZUxhbmRtYXJrTWFwcHBpbmdzOiBwb3NlTGFuZG1hcmtNYXBwaW5ncyxcbiAgICB9O1xuXG4gICAgcmV0dXJuIEpTT04uc3RyaW5naWZ5KGpzb24pO1xuICB9XG5cbiAgbG9hZEpzb24oanNvbjogc3RyaW5nIHwgYW55KSB7XG4gICAgY29uc3QgcGFyc2VkSnNvbiA9IHR5cGVvZiBqc29uID09PSAnc3RyaW5nJyA/IEpTT04ucGFyc2UoanNvbikgOiBqc29uO1xuXG4gICAgaWYgKHBhcnNlZEpzb24uZ2VuZXJhdG9yICE9PSAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InKSB7XG4gICAgICB0aHJvdyAn5LiN5q2j44Gq44OV44Kh44Kk44OrJztcbiAgICB9IGVsc2UgaWYgKHBhcnNlZEpzb24udmVyc2lvbiAhPT0gMSkge1xuICAgICAgdGhyb3cgJ+acquWvvuW/nOOBruODkOODvOOCuOODp+ODsyc7XG4gICAgfVxuXG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0gcGFyc2VkSnNvbi52aWRlbztcbiAgICB0aGlzLnBvc2VzID0gcGFyc2VkSnNvbi5wb3Nlcy5tYXAoKGl0ZW06IFBvc2VTZXRKc29uSXRlbSk6IFBvc2VTZXRJdGVtID0+IHtcbiAgICAgIGNvbnN0IHBvc2VWZWN0b3I6IGFueSA9IHt9O1xuICAgICAgUG9zZVNldC5QT1NFX1ZFQ1RPUl9NQVBQSU5HUy5tYXAoKGtleSwgaW5kZXgpID0+IHtcbiAgICAgICAgcG9zZVZlY3RvcltrZXkgYXMga2V5b2YgUG9zZVZlY3Rvcl0gPSBpdGVtLnZbaW5kZXhdO1xuICAgICAgfSk7XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIHZlY3RvcnM6IHBvc2VWZWN0b3IsXG4gICAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiB1bmRlZmluZWQsXG4gICAgICB9O1xuICAgIH0pO1xuICB9XG5cbiAgYXN5bmMgbG9hZFppcChidWZmZXI6IEFycmF5QnVmZmVyLCBpbmNsdWRlSW1hZ2VzOiBib29sZWFuID0gdHJ1ZSkge1xuICAgIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gbG9hZFppcC4uLmAsIEpTWmlwKTtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gaW5pdC4uLmApO1xuICAgIGNvbnN0IHppcCA9IGF3YWl0IGpzWmlwLmxvYWRBc3luYyhidWZmZXIsIHsgYmFzZTY0OiBmYWxzZSB9KTtcbiAgICBpZiAoIXppcCkgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OCkuiqreOBv+i+vOOCgeOBvuOBm+OCk+OBp+OBl+OBnyc7XG5cbiAgICBjb25zdCBqc29uID0gYXdhaXQgemlwLmZpbGUoJ3Bvc2VzLmpzb24nKT8uYXN5bmMoJ3RleHQnKTtcbiAgICBpZiAoanNvbiA9PT0gdW5kZWZpbmVkKSB7XG4gICAgICB0aHJvdyAnWklQ44OV44Kh44Kk44Or44GrIHBvc2UuanNvbiDjgYzlkKvjgb7jgozjgabjgYTjgb7jgZvjgpMnO1xuICAgIH1cblxuICAgIHRoaXMubG9hZEpzb24oanNvbik7XG5cbiAgICBjb25zdCBmaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBpZiAoaW5jbHVkZUltYWdlcykge1xuICAgICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc3QgZnJhbWVJbWFnZUZpbGVOYW1lID0gYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7ZmlsZUV4dH1gO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShmcmFtZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgICAgaWYgKCFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBwb3NlSW1hZ2VGaWxlTmFtZSA9IGBwb3NlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7ZmlsZUV4dH1gO1xuICAgICAgICAgIGNvbnN0IGltYWdlQmFzZTY0ID0gYXdhaXQgemlwXG4gICAgICAgICAgICAuZmlsZShwb3NlSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gYGRhdGE6JHt0aGlzLklNQUdFX01JTUV9O2Jhc2U2NCwke2ltYWdlQmFzZTY0fWA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuICB9XG59XG4iXX0=