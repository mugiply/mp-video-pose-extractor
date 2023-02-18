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
        const bodyVector = PoseSet.getBodyVector(poseLandmarksWithWorldCoordinate);
        if (!bodyVector) {
            console.warn(`[PoseSet] pushPose - Could not get the body vector`, poseLandmarksWithWorldCoordinate);
            return;
        }
        const handVector = PoseSet.getHandVectors(results.leftHandLandmarks, results.rightHandLandmarks);
        if (!handVector) {
            console.warn(`[PoseSet] pushPose - Could not get the hand vector`, results);
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
            bodyVectors: bodyVector,
            handVectors: handVector,
            frameImageDataUrl: frameImageDataUrl,
            poseImageDataUrl: poseImageDataUrl,
        };
        if (1 <= this.poses.length) {
            // 前回のポーズとの類似性をチェック
            const lastPose = this.poses[this.poses.length - 1];
            const isSimilarBodyPose = PoseSet.isSimilarBodyPose(lastPose.bodyVectors, pose.bodyVectors);
            const isSimilarHandPose = lastPose.handVectors && pose.handVectors
                ? PoseSet.isSimilarHandPose(lastPose.handVectors, pose.handVectors)
                : true;
            if (isSimilarBodyPose && isSimilarHandPose) {
                // 身体・手ともに類似ポーズならばスキップ
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
                const isSimilarBodyPose = PoseSet.isSimilarBodyPose(poseA.bodyVectors, poseB.bodyVectors);
                const isSimilarHandPose = poseA.handVectors && poseB.handVectors
                    ? PoseSet.isSimilarHandPose(poseA.handVectors, poseB.handVectors)
                    : true;
                if (isSimilarBodyPose && isSimilarHandPose) {
                    // 身体・手ともに類似ポーズならば
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
        const bodyVector = PoseSet.getBodyVector(results.ea);
        if (!bodyVector)
            throw 'Could not get the body vector';
        const poses = [];
        for (const pose of this.poses) {
            const similarity = PoseSet.getBodyPoseSimilarity(pose.bodyVectors, bodyVector);
            if (threshold <= similarity) {
                poses.push({
                    ...pose,
                    similarity: similarity,
                });
            }
        }
        return poses;
    }
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
    static getHandVectors(leftHandLandmarks, rightHandLandmarks) {
        return {
            // 右手 - 親指
            rightThumbTipToFirstJoint: [
                rightHandLandmarks[4].x - rightHandLandmarks[3].x,
                rightHandLandmarks[4].y - rightHandLandmarks[3].y,
                rightHandLandmarks[4].z - rightHandLandmarks[3].z,
            ],
            rightThumbFirstJointToSecondJoint: [
                rightHandLandmarks[3].x - rightHandLandmarks[2].x,
                rightHandLandmarks[3].y - rightHandLandmarks[2].y,
                rightHandLandmarks[3].z - rightHandLandmarks[2].z,
            ],
            // 右手 - 人差し指
            rightIndexFingerTipToFirstJoint: [
                rightHandLandmarks[8].x - rightHandLandmarks[7].x,
                rightHandLandmarks[8].y - rightHandLandmarks[7].y,
                rightHandLandmarks[8].z - rightHandLandmarks[7].z,
            ],
            rightIndexFingerFirstJointToSecondJoint: [
                rightHandLandmarks[7].x - rightHandLandmarks[6].x,
                rightHandLandmarks[7].y - rightHandLandmarks[6].y,
                rightHandLandmarks[7].z - rightHandLandmarks[6].z,
            ],
            // 右手 - 中指
            rightMiddleFingerTipToFirstJoint: [
                rightHandLandmarks[12].x - rightHandLandmarks[11].x,
                rightHandLandmarks[12].y - rightHandLandmarks[11].y,
                rightHandLandmarks[12].z - rightHandLandmarks[11].z,
            ],
            rightMiddleFingerFirstJointToSecondJoint: [
                rightHandLandmarks[11].x - rightHandLandmarks[10].x,
                rightHandLandmarks[11].y - rightHandLandmarks[10].y,
                rightHandLandmarks[11].z - rightHandLandmarks[10].z,
            ],
            // 右手 - 薬指
            rightRingFingerTipToFirstJoint: [
                rightHandLandmarks[16].x - rightHandLandmarks[15].x,
                rightHandLandmarks[16].y - rightHandLandmarks[15].y,
                rightHandLandmarks[16].z - rightHandLandmarks[15].z,
            ],
            rightRingFingerFirstJointToSecondJoint: [
                rightHandLandmarks[15].x - rightHandLandmarks[14].x,
                rightHandLandmarks[15].y - rightHandLandmarks[14].y,
                rightHandLandmarks[15].z - rightHandLandmarks[14].z,
            ],
            // 右手 - 小指
            rightPinkyFingerTipToFirstJoint: [
                rightHandLandmarks[20].x - rightHandLandmarks[19].x,
                rightHandLandmarks[20].y - rightHandLandmarks[19].y,
                rightHandLandmarks[20].z - rightHandLandmarks[19].z,
            ],
            rightPinkyFingerFirstJointToSecondJoint: [
                rightHandLandmarks[19].x - rightHandLandmarks[18].x,
                rightHandLandmarks[19].y - rightHandLandmarks[18].y,
                rightHandLandmarks[19].z - rightHandLandmarks[18].z,
            ],
            // 左手 - 親指
            leftThumbTipToFirstJoint: [
                leftHandLandmarks[4].x - leftHandLandmarks[3].x,
                leftHandLandmarks[4].y - leftHandLandmarks[3].y,
                leftHandLandmarks[4].z - leftHandLandmarks[3].z,
            ],
            leftThumbFirstJointToSecondJoint: [
                leftHandLandmarks[3].x - leftHandLandmarks[2].x,
                leftHandLandmarks[3].y - leftHandLandmarks[2].y,
                leftHandLandmarks[3].z - leftHandLandmarks[2].z,
            ],
            // 左手 - 人差し指
            leftIndexFingerTipToFirstJoint: [
                leftHandLandmarks[8].x - leftHandLandmarks[7].x,
                leftHandLandmarks[8].y - leftHandLandmarks[7].y,
                leftHandLandmarks[8].z - leftHandLandmarks[7].z,
            ],
            leftIndexFingerFirstJointToSecondJoint: [
                leftHandLandmarks[7].x - leftHandLandmarks[6].x,
                leftHandLandmarks[7].y - leftHandLandmarks[6].y,
                leftHandLandmarks[7].z - leftHandLandmarks[6].z,
            ],
            // 左手 - 中指
            leftMiddleFingerTipToFirstJoint: [
                leftHandLandmarks[12].x - leftHandLandmarks[11].x,
                leftHandLandmarks[12].y - leftHandLandmarks[11].y,
                leftHandLandmarks[12].z - leftHandLandmarks[11].z,
            ],
            leftMiddleFingerFirstJointToSecondJoint: [
                leftHandLandmarks[11].x - leftHandLandmarks[10].x,
                leftHandLandmarks[11].y - leftHandLandmarks[10].y,
                leftHandLandmarks[11].z - leftHandLandmarks[10].z,
            ],
            // 左手 - 薬指
            leftRingFingerTipToFirstJoint: [
                leftHandLandmarks[16].x - leftHandLandmarks[15].x,
                leftHandLandmarks[16].y - leftHandLandmarks[15].y,
                leftHandLandmarks[16].z - leftHandLandmarks[15].z,
            ],
            leftRingFingerFirstJointToSecondJoint: [
                leftHandLandmarks[15].x - leftHandLandmarks[14].x,
                leftHandLandmarks[15].y - leftHandLandmarks[14].y,
                leftHandLandmarks[15].z - leftHandLandmarks[14].z,
            ],
            // 左手 - 小指
            leftPinkyFingerTipToFirstJoint: [
                leftHandLandmarks[20].x - leftHandLandmarks[19].x,
                leftHandLandmarks[20].y - leftHandLandmarks[19].y,
                leftHandLandmarks[20].z - leftHandLandmarks[19].z,
            ],
            leftPinkyFingerFirstJointToSecondJoint: [
                leftHandLandmarks[19].x - leftHandLandmarks[18].x,
                leftHandLandmarks[19].y - leftHandLandmarks[18].y,
                leftHandLandmarks[19].z - leftHandLandmarks[18].z,
            ],
        };
    }
    static isSimilarBodyPose(bodyVectorA, bodyVectorB, threshold = 0.9) {
        let isSimilar = false;
        const similarity = PoseSet.getBodyPoseSimilarity(bodyVectorA, bodyVectorB);
        if (similarity >= threshold)
            isSimilar = true;
        // console.log(`[PoseSet] isSimilarPose`, isSimilar, similarity);
        return isSimilar;
    }
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
    static isSimilarHandPose(handVectorA, handVectorB, threshold = 0.9) {
        const similarity = PoseSet.getHandSimilarity(handVectorA, handVectorB);
        return similarity >= threshold;
    }
    static getHandSimilarity(handVectorA, handVectorB) {
        const cosSimilarities = {
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
                // BodyVector の圧縮
                const bodyVector = [];
                for (const key of PoseSet.BODY_VECTOR_MAPPINGS) {
                    bodyVector.push(pose.bodyVectors[key]);
                }
                // HandVector の圧縮
                let handVector = undefined;
                if (pose.handVectors) {
                    handVector = [];
                    for (const key of PoseSet.HAND_VECTOR_MAPPINGS) {
                        handVector.push(pose.handVectors[key]);
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
                bodyVectors: bodyVector,
                handVectors: handVector,
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
    // 右足
    'rightAnkleToRightKnee',
    'rightKneeToRightHip',
    // 左足
    'leftAnkleToLeftKnee',
    'leftKneeToLeftHip',
    // 胴体
    'rightHipToLeftHip',
    'rightShoulderToLeftShoulder',
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBaUZsQjtRQXZFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQXVEckMsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxxQkFBcUIsRUFBRSxDQUFDO1NBQ3pCLENBQUM7SUFDSixDQUFDO0lBRUQsWUFBWTtRQUNWLE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUM7SUFDakMsQ0FBQztJQUVELFlBQVksQ0FBQyxTQUFpQjtRQUM1QixJQUFJ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SUFDWixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDVixrQ0FBa0M7SUFDbEMsMENBQTBDO0lBQzFDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwwQkFBMEI7SUFDMUIsa0NBQWtDO0lBQ2xDLFlBQVk7SUFDWixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwrQkFBK0I7SUFDL0IsdUNBQXVDO0lBQ3ZDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLEtBQUs7SUFDTCx1QkFBdUI7SUFDdkIscUJBQXFCO0lBQ3JCLEtBQUs7SUFDTCxxQkFBcUI7SUFDckIsbUJBQW1CO0lBQ25CLEtBQUs7SUFDTCxtQkFBbUI7SUFDbkIsNkJBQTZCO0NBQzlCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBQT1NFX0xBTkRNQVJLUywgUmVzdWx0cyB9IGZyb20gJ0BtZWRpYXBpcGUvaG9saXN0aWMnO1xuaW1wb3J0ICogYXMgSlNaaXAgZnJvbSAnanN6aXAnO1xuaW1wb3J0IHsgUG9zZVNldEl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWl0ZW0nO1xuaW1wb3J0IHsgUG9zZVNldEpzb24gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24nO1xuaW1wb3J0IHsgUG9zZVNldEpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uLWl0ZW0nO1xuaW1wb3J0IHsgQm9keVZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvYm9keS12ZWN0b3InO1xuXG4vLyBAdHMtaWdub3JlXG5pbXBvcnQgY29zU2ltaWxhcml0eSBmcm9tICdjb3Mtc2ltaWxhcml0eSc7XG5pbXBvcnQgeyBTaW1pbGFyUG9zZUl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL21hdGNoZWQtcG9zZS1pdGVtJztcbmltcG9ydCB7IEltYWdlVHJpbW1lciB9IGZyb20gJy4vaW50ZXJuYWxzL2ltYWdlLXRyaW1tZXInO1xuaW1wb3J0IHsgSGFuZFZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvaGFuZC12ZWN0b3InO1xuXG5leHBvcnQgY2xhc3MgUG9zZVNldCB7XG4gIHB1YmxpYyBnZW5lcmF0b3I/OiBzdHJpbmc7XG4gIHB1YmxpYyB2ZXJzaW9uPzogbnVtYmVyO1xuICBwcml2YXRlIHZpZGVvTWV0YWRhdGEhOiB7XG4gICAgbmFtZTogc3RyaW5nO1xuICAgIHdpZHRoOiBudW1iZXI7XG4gICAgaGVpZ2h0OiBudW1iZXI7XG4gICAgZHVyYXRpb246IG51bWJlcjtcbiAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IG51bWJlcjtcbiAgfTtcbiAgcHVibGljIHBvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gIHB1YmxpYyBpc0ZpbmFsaXplZD86IGJvb2xlYW4gPSBmYWxzZTtcblxuICAvLyBCb2R5VmVjdG9yIOOBruOCreODvOWQjVxuICBwdWJsaWMgc3RhdGljIHJlYWRvbmx5IEJPRFlfVkVDVE9SX01BUFBJTkdTID0gW1xuICAgIC8vIOWPs+iFlVxuICAgICdyaWdodFdyaXN0VG9SaWdodEVsYm93JyxcbiAgICAncmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcicsXG4gICAgLy8g5bem6IWVXG4gICAgJ2xlZnRXcmlzdFRvTGVmdEVsYm93JyxcbiAgICAnbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXInLFxuICBdO1xuXG4gIC8vIEhhbmRWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSEFORF9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgJ3JpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdyaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAncmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAncmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICdyaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAnbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICdsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICdsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z6LazXG4gICAgJ3JpZ2h0QW5rbGVUb1JpZ2h0S25lZScsXG4gICAgJ3JpZ2h0S25lZVRvUmlnaHRIaXAnLFxuICAgIC8vIOW3pui2s1xuICAgICdsZWZ0QW5rbGVUb0xlZnRLbmVlJyxcbiAgICAnbGVmdEtuZWVUb0xlZnRIaXAnLFxuICAgIC8vIOiDtOS9k1xuICAgICdyaWdodEhpcFRvTGVmdEhpcCcsXG4gICAgJ3JpZ2h0U2hvdWxkZXJUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgLy8g55S75YOP5pu444GN5Ye644GX5pmC44Gu6Kit5a6aXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEg6IG51bWJlciA9IDEwODA7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUlNRTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPVxuICAgICdpbWFnZS93ZWJwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9RVUFMSVRZID0gMC44O1xuXG4gIC8vIOeUu+WDj+OBruS9meeZvemZpOWOu1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUiA9ICcjMDAwMDAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTEQgPSA1MDtcblxuICAvLyDnlLvlg4/jga7og4zmma/oibLnva7mj5tcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SID0gJyMwMTZBRkQnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IgPSAnI0ZGRkZGRjAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTEQgPSAxMzA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgZ2V0UG9zZXMoKTogUG9zZVNldEl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgZ2V0UG9zZUJ5VGltZSh0aW1lTWlsaXNlY29uZHM6IG51bWJlcik6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UudGltZU1pbGlzZWNvbmRzID09PSB0aW1lTWlsaXNlY29uZHMpO1xuICB9XG5cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHBvc2VJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICB2aWRlb1dpZHRoOiBudW1iZXIsXG4gICAgdmlkZW9IZWlnaHQ6IG51bWJlcixcbiAgICB2aWRlb0R1cmF0aW9uOiBudW1iZXIsXG4gICAgcmVzdWx0czogUmVzdWx0c1xuICApIHtcbiAgICB0aGlzLnNldFZpZGVvTWV0YURhdGEodmlkZW9XaWR0aCwgdmlkZW9IZWlnaHQsIHZpZGVvRHVyYXRpb24pO1xuXG4gICAgaWYgKHJlc3VsdHMucG9zZUxhbmRtYXJrcyA9PT0gdW5kZWZpbmVkKSByZXR1cm47XG5cbiAgICBpZiAodGhpcy5wb3Nlcy5sZW5ndGggPT09IDApIHtcbiAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5maXJzdFBvc2VEZXRlY3RlZFRpbWUgPSB2aWRlb1RpbWVNaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICBjb25zdCBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZTogYW55W10gPSAocmVzdWx0cyBhcyBhbnkpLmVhXG4gICAgICA/IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgIDogW107XG4gICAgaWYgKHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlLmxlbmd0aCA9PT0gMCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlIC0gQ291bGQgbm90IGdldCB0aGUgcG9zZSB3aXRoIHRoZSB3b3JsZCBjb29yZGluYXRlYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICBjb25zdCBib2R5VmVjdG9yID0gUG9zZVNldC5nZXRCb2R5VmVjdG9yKHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlKTtcbiAgICBpZiAoIWJvZHlWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAtIENvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yYCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGVcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgaGFuZFZlY3RvciA9IFBvc2VTZXQuZ2V0SGFuZFZlY3RvcnMoXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICApO1xuICAgIGlmICghaGFuZFZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3JgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2U6IFBvc2VTZXRJdGVtID0ge1xuICAgICAgdGltZU1pbGlzZWNvbmRzOiB2aWRlb1RpbWVNaWxpc2Vjb25kcyxcbiAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IC0xLFxuICAgICAgcG9zZTogcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubWFwKCh3b3JsZENvb3JkaW5hdGVMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLngsXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueSxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay56LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnZpc2liaWxpdHksXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGxlZnRIYW5kOiByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICByaWdodEhhbmQ6IHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGJvZHlWZWN0b3JzOiBib2R5VmVjdG9yLFxuICAgICAgaGFuZFZlY3RvcnM6IGhhbmRWZWN0b3IsXG4gICAgICBmcmFtZUltYWdlRGF0YVVybDogZnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBwb3NlSW1hZ2VEYXRhVXJsOiBwb3NlSW1hZ2VEYXRhVXJsLFxuICAgIH07XG5cbiAgICBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgLy8g5YmN5Zue44Gu44Od44O844K644Go44Gu6aGe5Ly85oCn44KS44OB44Kn44OD44KvXG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcblxuICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICBsYXN0UG9zZS5ib2R5VmVjdG9ycyxcbiAgICAgICAgcG9zZS5ib2R5VmVjdG9yc1xuICAgICAgKTtcbiAgICAgIGNvbnN0IGlzU2ltaWxhckhhbmRQb3NlID1cbiAgICAgICAgbGFzdFBvc2UuaGFuZFZlY3RvcnMgJiYgcG9zZS5oYW5kVmVjdG9yc1xuICAgICAgICAgID8gUG9zZVNldC5pc1NpbWlsYXJIYW5kUG9zZShsYXN0UG9zZS5oYW5kVmVjdG9ycywgcG9zZS5oYW5kVmVjdG9ycylcbiAgICAgICAgICA6IHRydWU7XG5cbiAgICAgIGlmIChpc1NpbWlsYXJCb2R5UG9zZSAmJiBpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAvLyDouqvkvZPjg7vmiYvjgajjgoLjgavpoZ7kvLzjg53jg7zjgrrjgarjgonjgbDjgrnjgq3jg4Pjg5dcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuXG4gICAgICAvLyDliY3lm57jga7jg53jg7zjgrrjga7mjIHntprmmYLplpPjgpLoqK3lrppcbiAgICAgIGNvbnN0IHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgdmlkZW9UaW1lTWlsaXNlY29uZHMgLSBsYXN0UG9zZS50aW1lTWlsaXNlY29uZHM7XG4gICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIHRoaXMucG9zZXMucHVzaChwb3NlKTtcbiAgfVxuXG4gIGFzeW5jIGZpbmFsaXplKCkge1xuICAgIGlmICgwID09IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICAvLyDmnIDlvozjga7jg53jg7zjgrrjga7mjIHntprmmYLplpPjgpLoqK3lrppcbiAgICBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgY29uc3QgbGFzdFBvc2UgPSB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV07XG4gICAgICBpZiAobGFzdFBvc2UuZHVyYXRpb25NaWxpc2Vjb25kcyA9PSAtMSkge1xuICAgICAgICBjb25zdCBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgICAgdGhpcy52aWRlb01ldGFkYXRhLmR1cmF0aW9uIC0gbGFzdFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgICAgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8g6YeN6KSH44Od44O844K644KS6Zmk5Y67XG4gICAgdGhpcy5yZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTtcblxuICAgIC8vIOeUu+WDj+OBruODnuODvOOCuOODs+OCkuWPluW+l1xuICAgIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RpbmcgaW1hZ2UgbWFyZ2lucy4uLmApO1xuICAgIGxldCBpbWFnZVRyaW1taW5nOlxuICAgICAgfCB7XG4gICAgICAgICAgbWFyZ2luVG9wOiBudW1iZXI7XG4gICAgICAgICAgbWFyZ2luQm90dG9tOiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0TmV3OiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0T2xkOiBudW1iZXI7XG4gICAgICAgICAgd2lkdGg6IG51bWJlcjtcbiAgICAgICAgfVxuICAgICAgfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgY29uc3QgbWFyZ2luQ29sb3IgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0TWFyZ2luQ29sb3IoKTtcbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0ZWQgbWFyZ2luIGNvbG9yLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgIG1hcmdpbkNvbG9yXG4gICAgICApO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yID09PSBudWxsKSBjb250aW51ZTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciAhPT0gdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfQ09MT1IpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBjb25zdCB0cmltbWVkID0gYXdhaXQgaW1hZ2VUcmltbWVyLnRyaW1NYXJnaW4oXG4gICAgICAgIG1hcmdpbkNvbG9yLFxuICAgICAgICB0aGlzLklNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRFxuICAgICAgKTtcbiAgICAgIGlmICghdHJpbW1lZCkgY29udGludWU7XG4gICAgICBpbWFnZVRyaW1taW5nID0gdHJpbW1lZDtcbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZXJtaW5lZCBpbWFnZSB0cmltbWluZyBwb3NpdGlvbnMuLi5gLFxuICAgICAgICB0cmltbWVkXG4gICAgICApO1xuICAgICAgYnJlYWs7XG4gICAgfVxuXG4gICAgLy8g55S75YOP44KS5pW05b2iXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwgfHwgIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBQcm9jZXNzaW5nIGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHNcbiAgICAgICk7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODleODrOODvOODoOeUu+WDj1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVwbGFjZUNvbG9yKFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbGV0IG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZnJhbWUgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODneODvOOCuuODl+ODrOODk+ODpeODvOeUu+WDj1xuICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgaWYgKGltYWdlVHJpbW1pbmcpIHtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgICAgMCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLm1hcmdpblRvcCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLndpZHRoLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcuaGVpZ2h0TmV3XG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBwb3NlIHByZXZpZXcgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcbiAgICB9XG5cbiAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgfVxuXG4gIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcygpOiB2b2lkIHtcbiAgICAvLyDlhajjg53jg7zjgrrjgpLmr5TovIPjgZfjgabpoZ7kvLzjg53jg7zjgrrjgpLliYrpmaRcbiAgICBjb25zdCBuZXdQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZUEgb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGlzRHVwbGljYXRlZCA9IGZhbHNlO1xuICAgICAgZm9yIChjb25zdCBwb3NlQiBvZiBuZXdQb3Nlcykge1xuICAgICAgICBjb25zdCBpc1NpbWlsYXJCb2R5UG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgICAgICAgcG9zZUEuYm9keVZlY3RvcnMsXG4gICAgICAgICAgcG9zZUIuYm9keVZlY3RvcnNcbiAgICAgICAgKTtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFySGFuZFBvc2UgPVxuICAgICAgICAgIHBvc2VBLmhhbmRWZWN0b3JzICYmIHBvc2VCLmhhbmRWZWN0b3JzXG4gICAgICAgICAgICA/IFBvc2VTZXQuaXNTaW1pbGFySGFuZFBvc2UocG9zZUEuaGFuZFZlY3RvcnMsIHBvc2VCLmhhbmRWZWN0b3JzKVxuICAgICAgICAgICAgOiB0cnVlO1xuXG4gICAgICAgIGlmIChpc1NpbWlsYXJCb2R5UG9zZSAmJiBpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBqOOCguOBq+mhnuS8vOODneODvOOCuuOBquOCieOBsFxuICAgICAgICAgIGlzRHVwbGljYXRlZCA9IHRydWU7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKGlzRHVwbGljYXRlZCkgY29udGludWU7XG5cbiAgICAgIG5ld1Bvc2VzLnB1c2gocG9zZUEpO1xuICAgIH1cblxuICAgIGNvbnNvbGUuaW5mbyhcbiAgICAgIGBbUG9zZVNldF0gcmVtb3ZlRHVwbGljYXRlZFBvc2VzIC0gUmVkdWNlZCAke3RoaXMucG9zZXMubGVuZ3RofSBwb3NlcyAtPiAke25ld1Bvc2VzLmxlbmd0aH0gcG9zZXNgXG4gICAgKTtcbiAgICB0aGlzLnBvc2VzID0gbmV3UG9zZXM7XG4gIH1cblxuICBnZXRTaW1pbGFyUG9zZXMoXG4gICAgcmVzdWx0czogUmVzdWx0cyxcbiAgICB0aHJlc2hvbGQ6IG51bWJlciA9IDAuOVxuICApOiBTaW1pbGFyUG9zZUl0ZW1bXSB7XG4gICAgY29uc3QgYm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3RvcigocmVzdWx0cyBhcyBhbnkpLmVhKTtcbiAgICBpZiAoIWJvZHlWZWN0b3IpIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3Rvcic7XG5cbiAgICBjb25zdCBwb3NlcyA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgICAgIHBvc2UuYm9keVZlY3RvcnMsXG4gICAgICAgIGJvZHlWZWN0b3JcbiAgICAgICk7XG4gICAgICBpZiAodGhyZXNob2xkIDw9IHNpbWlsYXJpdHkpIHtcbiAgICAgICAgcG9zZXMucHVzaCh7XG4gICAgICAgICAgLi4ucG9zZSxcbiAgICAgICAgICBzaW1pbGFyaXR5OiBzaW1pbGFyaXR5LFxuICAgICAgICB9KTtcbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gcG9zZXM7XG4gIH1cblxuICBzdGF0aWMgZ2V0Qm9keVZlY3RvcihcbiAgICBwb3NlTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdXG4gICk6IEJvZHlWZWN0b3IgfCB1bmRlZmluZWQge1xuICAgIHJldHVybiB7XG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnosXG4gICAgICBdLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIHN0YXRpYyBnZXRIYW5kVmVjdG9ycyhcbiAgICBsZWZ0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXSxcbiAgICByaWdodEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogSGFuZFZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICAgcmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS54LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS56LFxuICAgICAgXSxcbiAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS54LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS56LFxuICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgcmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS54LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS56LFxuICAgICAgXSxcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS54LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS56LFxuICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFtcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS54LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueixcbiAgICAgIF0sXG4gICAgICByaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBbXG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueCxcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLnosXG4gICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgICByaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IFtcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54LFxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueixcbiAgICAgIF0sXG4gICAgICByaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLngsXG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS56LFxuICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTldLngsXG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS56LFxuICAgICAgXSxcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogW1xuICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnggLSByaWdodEhhbmRMYW5kbWFya3NbMThdLngsXG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS56LFxuICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICAgbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50OiBbXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1szXS54LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFtcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIGxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDogW1xuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLnksXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBbXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnggLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS54LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLnosXG4gICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgICBsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBbXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICBdLFxuICAgICAgbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBbXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLnksXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLnosXG4gICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g6Jas5oyHXG4gICAgICBsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDogW1xuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS54LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IFtcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAgIGxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogW1xuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS55LFxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBbXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLngsXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLnosXG4gICAgICBdLFxuICAgIH07XG4gIH1cblxuICBzdGF0aWMgaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC45XG4gICk6IGJvb2xlYW4ge1xuICAgIGxldCBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoYm9keVZlY3RvckEsIGJvZHlWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQpIGlzU2ltaWxhciA9IHRydWU7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGlzU2ltaWxhclBvc2VgLCBpc1NpbWlsYXIsIHNpbWlsYXJpdHkpO1xuXG4gICAgcmV0dXJuIGlzU2ltaWxhcjtcbiAgfVxuXG4gIHN0YXRpYyBnZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXMgPSB7XG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEubGVmdFdyaXN0VG9MZWZ0RWxib3csXG4gICAgICAgIGJvZHlWZWN0b3JCLmxlZnRXcmlzdFRvTGVmdEVsYm93XG4gICAgICApLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyLFxuICAgICAgICBib2R5VmVjdG9yQi5sZWZ0RWxib3dUb0xlZnRTaG91bGRlclxuICAgICAgKSxcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3csXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3dcbiAgICAgICksXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5yaWdodEVsYm93VG9SaWdodFNob3VsZGVyLFxuICAgICAgICBib2R5VmVjdG9yQi5yaWdodEVsYm93VG9SaWdodFNob3VsZGVyXG4gICAgICApLFxuICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNTdW0gPSBPYmplY3QudmFsdWVzKGNvc1NpbWlsYXJpdGllcykucmVkdWNlKFxuICAgICAgKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLFxuICAgICAgMFxuICAgICk7XG4gICAgcmV0dXJuIGNvc1NpbWlsYXJpdGllc1N1bSAvIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllcykubGVuZ3RoO1xuICB9XG5cbiAgc3RhdGljIGlzU2ltaWxhckhhbmRQb3NlKFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuOVxuICApOiBib29sZWFuIHtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShoYW5kVmVjdG9yQSwgaGFuZFZlY3RvckIpO1xuICAgIHJldHVybiBzaW1pbGFyaXR5ID49IHRocmVzaG9sZDtcbiAgfVxuXG4gIHN0YXRpYyBnZXRIYW5kU2ltaWxhcml0eShcbiAgICBoYW5kVmVjdG9yQTogSGFuZFZlY3RvcixcbiAgICBoYW5kVmVjdG9yQjogSGFuZFZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllcyA9IHtcbiAgICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICAgcmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgKSxcbiAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICksXG4gICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICksXG4gICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICApLFxuICAgICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgICByaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICApLFxuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIucmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgKSxcbiAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICApLFxuICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgKSxcbiAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgKSxcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICksXG4gICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICksXG4gICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICApLFxuICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICApLFxuICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgKSxcbiAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgKSxcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICksXG4gICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICksXG4gICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICApLFxuICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICApLFxuICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgKSxcbiAgICB9O1xuXG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzU3VtID0gT2JqZWN0LnZhbHVlcyhjb3NTaW1pbGFyaXRpZXMpLnJlZHVjZShcbiAgICAgIChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSxcbiAgICAgIDBcbiAgICApO1xuICAgIHJldHVybiBjb3NTaW1pbGFyaXRpZXNTdW0gLyBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXMpLmxlbmd0aDtcbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRaaXAoKTogUHJvbWlzZTxCbG9iPiB7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBqc1ppcC5maWxlKCdwb3Nlcy5qc29uJywgYXdhaXQgdGhpcy5nZXRKc29uKCkpO1xuXG4gICAgY29uc3QgaW1hZ2VGaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UucG9zZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICBnZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKElNQUdFX01JTUU6IHN0cmluZykge1xuICAgIHN3aXRjaCAoSU1BR0VfTUlNRSkge1xuICAgICAgY2FzZSAnaW1hZ2UvcG5nJzpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgICAgY2FzZSAnaW1hZ2UvanBlZyc6XG4gICAgICAgIHJldHVybiAnanBnJztcbiAgICAgIGNhc2UgJ2ltYWdlL3dlYnAnOlxuICAgICAgICByZXR1cm4gJ3dlYnAnO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgIH1cbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRKc29uKCk6IFByb21pc2U8c3RyaW5nPiB7XG4gICAgaWYgKHRoaXMudmlkZW9NZXRhZGF0YSA9PT0gdW5kZWZpbmVkIHx8IHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZClcbiAgICAgIHJldHVybiAne30nO1xuXG4gICAgaWYgKCF0aGlzLmlzRmluYWxpemVkKSB7XG4gICAgICBhd2FpdCB0aGlzLmZpbmFsaXplKCk7XG4gICAgfVxuXG4gICAgbGV0IHBvc2VMYW5kbWFya01hcHBpbmdzID0gW107XG4gICAgZm9yIChjb25zdCBrZXkgb2YgT2JqZWN0LmtleXMoUE9TRV9MQU5ETUFSS1MpKSB7XG4gICAgICBjb25zdCBpbmRleDogbnVtYmVyID0gUE9TRV9MQU5ETUFSS1Nba2V5IGFzIGtleW9mIHR5cGVvZiBQT1NFX0xBTkRNQVJLU107XG4gICAgICBwb3NlTGFuZG1hcmtNYXBwaW5nc1tpbmRleF0gPSBrZXk7XG4gICAgfVxuXG4gICAgY29uc3QganNvbjogUG9zZVNldEpzb24gPSB7XG4gICAgICBnZW5lcmF0b3I6ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicsXG4gICAgICB2ZXJzaW9uOiAxLFxuICAgICAgdmlkZW86IHRoaXMudmlkZW9NZXRhZGF0YSEsXG4gICAgICBwb3NlczogdGhpcy5wb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKTogUG9zZVNldEpzb25JdGVtID0+IHtcbiAgICAgICAgLy8gQm9keVZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgY29uc3QgYm9keVZlY3RvciA9IFtdO1xuICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgYm9keVZlY3Rvci5wdXNoKHBvc2UuYm9keVZlY3RvcnNba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIEhhbmRWZWN0b3Ig44Gu5Zyn57iuXG4gICAgICAgIGxldCBoYW5kVmVjdG9yOiBudW1iZXJbXVtdIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgICAgICBpZiAocG9zZS5oYW5kVmVjdG9ycykge1xuICAgICAgICAgIGhhbmRWZWN0b3IgPSBbXTtcbiAgICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkhBTkRfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgICBoYW5kVmVjdG9yLnB1c2gocG9zZS5oYW5kVmVjdG9yc1trZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0pO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIC8vIFBvc2VTZXRKc29uSXRlbSDjga4gcG9zZSDjgqrjg5bjgrjjgqfjgq/jg4jjgpLnlJ/miJBcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICB0OiBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgICBkOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgcDogcG9zZS5wb3NlLFxuICAgICAgICAgIGw6IHBvc2UubGVmdEhhbmQsXG4gICAgICAgICAgcjogcG9zZS5yaWdodEhhbmQsXG4gICAgICAgICAgdjogYm9keVZlY3RvcixcbiAgICAgICAgICBoOiBoYW5kVmVjdG9yLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcCgoaXRlbTogUG9zZVNldEpzb25JdGVtKTogUG9zZVNldEl0ZW0gPT4ge1xuICAgICAgY29uc3QgYm9keVZlY3RvcjogYW55ID0ge307XG4gICAgICBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICBib2R5VmVjdG9yW2tleSBhcyBrZXlvZiBCb2R5VmVjdG9yXSA9IGl0ZW0udltpbmRleF07XG4gICAgICB9KTtcblxuICAgICAgY29uc3QgaGFuZFZlY3RvcjogYW55ID0ge307XG4gICAgICBpZiAoaXRlbS5oKSB7XG4gICAgICAgIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgICAgaGFuZFZlY3RvcltrZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0gPSBpdGVtLmghW2luZGV4XTtcbiAgICAgICAgfSk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIGJvZHlWZWN0b3JzOiBib2R5VmVjdG9yLFxuICAgICAgICBoYW5kVmVjdG9yczogaGFuZFZlY3RvcixcbiAgICAgICAgZnJhbWVJbWFnZURhdGFVcmw6IHVuZGVmaW5lZCxcbiAgICAgIH07XG4gICAgfSk7XG4gIH1cblxuICBhc3luYyBsb2FkWmlwKGJ1ZmZlcjogQXJyYXlCdWZmZXIsIGluY2x1ZGVJbWFnZXM6IGJvb2xlYW4gPSB0cnVlKSB7XG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBsb2FkWmlwLi4uYCwgSlNaaXApO1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBpbml0Li4uYCk7XG4gICAgY29uc3QgemlwID0gYXdhaXQganNaaXAubG9hZEFzeW5jKGJ1ZmZlciwgeyBiYXNlNjQ6IGZhbHNlIH0pO1xuICAgIGlmICghemlwKSB0aHJvdyAnWklQ44OV44Kh44Kk44Or44KS6Kqt44G/6L6844KB44G+44Gb44KT44Gn44GX44GfJztcblxuICAgIGNvbnN0IGpzb24gPSBhd2FpdCB6aXAuZmlsZSgncG9zZXMuanNvbicpPy5hc3luYygndGV4dCcpO1xuICAgIGlmIChqc29uID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgasgcG9zZS5qc29uIOOBjOWQq+OBvuOCjOOBpuOBhOOBvuOBm+OCkyc7XG4gICAgfVxuXG4gICAgdGhpcy5sb2FkSnNvbihqc29uKTtcblxuICAgIGNvbnN0IGZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGlmIChpbmNsdWRlSW1hZ2VzKSB7XG4gICAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBmcmFtZUltYWdlRmlsZU5hbWUgPSBgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKGZyYW1lSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IHBvc2VJbWFnZUZpbGVOYW1lID0gYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKHBvc2VJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cbiJdfQ==