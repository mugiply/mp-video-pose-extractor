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
    pushPose(videoTimeMiliseconds, frameImageDataUrl, poseImageDataUrl, faceFrameImageDataUrl, results) {
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
        const handVector = PoseSet.getHandVectors(results.leftHandLandmarks, results.rightHandLandmarks);
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
            faceFrameImageDataUrl: faceFrameImageDataUrl,
            extendedData: {},
        };
        if (1 <= this.poses.length) {
            // 前回のポーズとの類似性をチェック
            const lastPose = this.poses[this.poses.length - 1];
            const isSimilarBodyPose = PoseSet.isSimilarBodyPose(lastPose.bodyVectors, pose.bodyVectors);
            let isSimilarHandPose = true;
            if (lastPose.handVectors && pose.handVectors) {
                isSimilarHandPose = PoseSet.isSimilarHandPose(lastPose.handVectors, pose.handVectors);
            }
            else if (!lastPose.handVectors && pose.handVectors) {
                isSimilarHandPose = false;
            }
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
        return pose;
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
        // 最初のポーズを除去
        this.poses.shift();
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
            if (pose.faceFrameImageDataUrl) {
                // 画像を整形 - 顔フレーム画像
                imageTrimmer = new ImageTrimmer();
                await imageTrimmer.loadByDataUrl(pose.faceFrameImageDataUrl);
                newDataUrl = await imageTrimmer.getDataUrl(this.IMAGE_MIME, this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
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
                    : false;
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
    static isSimilarBodyPose(bodyVectorA, bodyVectorB, threshold = 0.8) {
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
    static isSimilarHandPose(handVectorA, handVectorB, threshold = 0.75) {
        const similarity = PoseSet.getHandSimilarity(handVectorA, handVectorB);
        if (similarity === -1) {
            return true;
        }
        return similarity >= threshold;
    }
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
        let cosSimilaritiesSumLeftHand = 0;
        if (cosSimilaritiesLeftHand) {
            cosSimilaritiesSumLeftHand = Object.values(cosSimilaritiesLeftHand).reduce((sum, value) => sum + value, 0);
        }
        let cosSimilaritiesSumRightHand = 0;
        if (cosSimilaritiesRightHand) {
            cosSimilaritiesSumRightHand = Object.values(cosSimilaritiesRightHand).reduce((sum, value) => sum + value, 0);
        }
        if (cosSimilaritiesRightHand && cosSimilaritiesLeftHand) {
            return ((cosSimilaritiesSumRightHand + cosSimilaritiesSumLeftHand) /
                (Object.keys(cosSimilaritiesRightHand).length +
                    Object.keys(cosSimilaritiesLeftHand).length));
        }
        else if (cosSimilaritiesSumRightHand) {
            return (cosSimilaritiesSumRightHand /
                Object.keys(cosSimilaritiesRightHand).length);
        }
        else if (cosSimilaritiesLeftHand) {
            return (cosSimilaritiesSumLeftHand /
                Object.keys(cosSimilaritiesLeftHand).length);
        }
        else {
            return -1;
        }
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
                    e: pose.extendedData,
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
                extendedData: item.e,
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBaUZsQjtRQXZFTyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQXVEckMsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxxQkFBcUIsRUFBRSxDQUFDO1NBQ3pCLENBQUM7SUFDSixDQUFDO0lBRUQsWUFBWTtRQUNWLE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUM7SUFDakMsQ0FBQztJQUVELFlBQVksQ0FBQyxTQUFpQjtRQUM1QixJQUFJ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QVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwwQkFBMEI7SUFDMUIsa0NBQWtDO0lBQ2xDLFlBQVk7SUFDWixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwrQkFBK0I7SUFDL0IsdUNBQXVDO0lBQ3ZDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLEtBQUs7SUFDTCx1QkFBdUI7SUFDdkIscUJBQXFCO0lBQ3JCLEtBQUs7SUFDTCxxQkFBcUI7SUFDckIsbUJBQW1CO0lBQ25CLEtBQUs7SUFDTCxtQkFBbUI7SUFDbkIsNkJBQTZCO0NBQzlCLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBQT1NFX0xBTkRNQVJLUywgUmVzdWx0cyB9IGZyb20gJ0BtZWRpYXBpcGUvaG9saXN0aWMnO1xuaW1wb3J0ICogYXMgSlNaaXAgZnJvbSAnanN6aXAnO1xuaW1wb3J0IHsgUG9zZVNldEl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWl0ZW0nO1xuaW1wb3J0IHsgUG9zZVNldEpzb24gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24nO1xuaW1wb3J0IHsgUG9zZVNldEpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uLWl0ZW0nO1xuaW1wb3J0IHsgQm9keVZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvYm9keS12ZWN0b3InO1xuXG4vLyBAdHMtaWdub3JlXG5pbXBvcnQgY29zU2ltaWxhcml0eSBmcm9tICdjb3Mtc2ltaWxhcml0eSc7XG5pbXBvcnQgeyBTaW1pbGFyUG9zZUl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3NpbWlsYXItcG9zZS1pdGVtJztcbmltcG9ydCB7IEltYWdlVHJpbW1lciB9IGZyb20gJy4vaW50ZXJuYWxzL2ltYWdlLXRyaW1tZXInO1xuaW1wb3J0IHsgSGFuZFZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvaGFuZC12ZWN0b3InO1xuXG5leHBvcnQgY2xhc3MgUG9zZVNldCB7XG4gIHB1YmxpYyBnZW5lcmF0b3I/OiBzdHJpbmc7XG4gIHB1YmxpYyB2ZXJzaW9uPzogbnVtYmVyO1xuICBwcml2YXRlIHZpZGVvTWV0YWRhdGEhOiB7XG4gICAgbmFtZTogc3RyaW5nO1xuICAgIHdpZHRoOiBudW1iZXI7XG4gICAgaGVpZ2h0OiBudW1iZXI7XG4gICAgZHVyYXRpb246IG51bWJlcjtcbiAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IG51bWJlcjtcbiAgfTtcbiAgcHVibGljIHBvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gIHB1YmxpYyBpc0ZpbmFsaXplZD86IGJvb2xlYW4gPSBmYWxzZTtcblxuICAvLyBCb2R5VmVjdG9yIOOBruOCreODvOWQjVxuICBwdWJsaWMgc3RhdGljIHJlYWRvbmx5IEJPRFlfVkVDVE9SX01BUFBJTkdTID0gW1xuICAgIC8vIOWPs+iFlVxuICAgICdyaWdodFdyaXN0VG9SaWdodEVsYm93JyxcbiAgICAncmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcicsXG4gICAgLy8g5bem6IWVXG4gICAgJ2xlZnRXcmlzdFRvTGVmdEVsYm93JyxcbiAgICAnbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXInLFxuICBdO1xuXG4gIC8vIEhhbmRWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSEFORF9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgJ3JpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdyaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAncmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAncmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICdyaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAnbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICdsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICdsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5Y+z6LazXG4gICAgJ3JpZ2h0QW5rbGVUb1JpZ2h0S25lZScsXG4gICAgJ3JpZ2h0S25lZVRvUmlnaHRIaXAnLFxuICAgIC8vIOW3pui2s1xuICAgICdsZWZ0QW5rbGVUb0xlZnRLbmVlJyxcbiAgICAnbGVmdEtuZWVUb0xlZnRIaXAnLFxuICAgIC8vIOiDtOS9k1xuICAgICdyaWdodEhpcFRvTGVmdEhpcCcsXG4gICAgJ3JpZ2h0U2hvdWxkZXJUb0xlZnRTaG91bGRlcicsXG4gIF07XG5cbiAgLy8g55S75YOP5pu444GN5Ye644GX5pmC44Gu6Kit5a6aXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEg6IG51bWJlciA9IDEwODA7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUlNRTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPVxuICAgICdpbWFnZS93ZWJwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9RVUFMSVRZID0gMC44O1xuXG4gIC8vIOeUu+WDj+OBruS9meeZvemZpOWOu1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUiA9ICcjMDAwMDAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTEQgPSA1MDtcblxuICAvLyDnlLvlg4/jga7og4zmma/oibLnva7mj5tcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SID0gJyMwMTZBRkQnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IgPSAnI0ZGRkZGRjAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTEQgPSAxMzA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgZ2V0UG9zZXMoKTogUG9zZVNldEl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgZ2V0UG9zZUJ5VGltZSh0aW1lTWlsaXNlY29uZHM6IG51bWJlcik6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UudGltZU1pbGlzZWNvbmRzID09PSB0aW1lTWlsaXNlY29uZHMpO1xuICB9XG5cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHBvc2VJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICk6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHJldHVybjtcblxuICAgIGlmICh0aGlzLnBvc2VzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhpcy52aWRlb01ldGFkYXRhLmZpcnN0UG9zZURldGVjdGVkVGltZSA9IHZpZGVvVGltZU1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHBvc2Ugd2l0aCB0aGUgd29ybGQgY29vcmRpbmF0ZWAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgYm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3Rvcihwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZSk7XG4gICAgaWYgKCFib2R5VmVjdG9yKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yYCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGVcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgaWYgKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkICYmXG4gICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkXG4gICAgKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBsZWZ0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgcmlnaHQgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IGhhbmRWZWN0b3IgPSBQb3NlU2V0LmdldEhhbmRWZWN0b3JzKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgIHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzXG4gICAgKTtcbiAgICBpZiAoIWhhbmRWZWN0b3IpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3JgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2U6IFBvc2VTZXRJdGVtID0ge1xuICAgICAgdGltZU1pbGlzZWNvbmRzOiB2aWRlb1RpbWVNaWxpc2Vjb25kcyxcbiAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IC0xLFxuICAgICAgcG9zZTogcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubWFwKCh3b3JsZENvb3JkaW5hdGVMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLngsXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueSxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay56LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnZpc2liaWxpdHksXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGxlZnRIYW5kOiByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzPy5tYXAoKG5vcm1hbGl6ZWRMYW5kbWFyaykgPT4ge1xuICAgICAgICByZXR1cm4gW1xuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay54LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay55LFxuICAgICAgICAgIG5vcm1hbGl6ZWRMYW5kbWFyay56LFxuICAgICAgICBdO1xuICAgICAgfSksXG4gICAgICByaWdodEhhbmQ6IHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGJvZHlWZWN0b3JzOiBib2R5VmVjdG9yLFxuICAgICAgaGFuZFZlY3RvcnM6IGhhbmRWZWN0b3IsXG4gICAgICBmcmFtZUltYWdlRGF0YVVybDogZnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBwb3NlSW1hZ2VEYXRhVXJsOiBwb3NlSW1hZ2VEYXRhVXJsLFxuICAgICAgZmFjZUZyYW1lSW1hZ2VEYXRhVXJsOiBmYWNlRnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBleHRlbmRlZERhdGE6IHt9LFxuICAgIH07XG5cbiAgICBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgLy8g5YmN5Zue44Gu44Od44O844K644Go44Gu6aGe5Ly85oCn44KS44OB44Kn44OD44KvXG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcblxuICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICBsYXN0UG9zZS5ib2R5VmVjdG9ycyxcbiAgICAgICAgcG9zZS5ib2R5VmVjdG9yc1xuICAgICAgKTtcblxuICAgICAgbGV0IGlzU2ltaWxhckhhbmRQb3NlID0gdHJ1ZTtcbiAgICAgIGlmIChsYXN0UG9zZS5oYW5kVmVjdG9ycyAmJiBwb3NlLmhhbmRWZWN0b3JzKSB7XG4gICAgICAgIGlzU2ltaWxhckhhbmRQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJIYW5kUG9zZShcbiAgICAgICAgICBsYXN0UG9zZS5oYW5kVmVjdG9ycyxcbiAgICAgICAgICBwb3NlLmhhbmRWZWN0b3JzXG4gICAgICAgICk7XG4gICAgICB9IGVsc2UgaWYgKCFsYXN0UG9zZS5oYW5kVmVjdG9ycyAmJiBwb3NlLmhhbmRWZWN0b3JzKSB7XG4gICAgICAgIGlzU2ltaWxhckhhbmRQb3NlID0gZmFsc2U7XG4gICAgICB9XG5cbiAgICAgIGlmIChpc1NpbWlsYXJCb2R5UG9zZSAmJiBpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAvLyDouqvkvZPjg7vmiYvjgajjgoLjgavpoZ7kvLzjg53jg7zjgrrjgarjgonjgbDjgrnjgq3jg4Pjg5dcbiAgICAgICAgcmV0dXJuO1xuICAgICAgfVxuXG4gICAgICAvLyDliY3lm57jga7jg53jg7zjgrrjga7mjIHntprmmYLplpPjgpLoqK3lrppcbiAgICAgIGNvbnN0IHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgdmlkZW9UaW1lTWlsaXNlY29uZHMgLSBsYXN0UG9zZS50aW1lTWlsaXNlY29uZHM7XG4gICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIHRoaXMucG9zZXMucHVzaChwb3NlKTtcblxuICAgIHJldHVybiBwb3NlO1xuICB9XG5cbiAgYXN5bmMgZmluYWxpemUoKSB7XG4gICAgaWYgKDAgPT0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIHRoaXMuaXNGaW5hbGl6ZWQgPSB0cnVlO1xuICAgICAgcmV0dXJuO1xuICAgIH1cblxuICAgIC8vIOacgOW+jOOBruODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgIGlmICgxIDw9IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcbiAgICAgIGlmIChsYXN0UG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzID09IC0xKSB7XG4gICAgICAgIGNvbnN0IHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gLSBsYXN0UG9zZS50aW1lTWlsaXNlY29uZHM7XG4gICAgICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgICBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcztcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDph43opIfjg53jg7zjgrrjgpLpmaTljrtcbiAgICB0aGlzLnJlbW92ZUR1cGxpY2F0ZWRQb3NlcygpO1xuXG4gICAgLy8g5pyA5Yid44Gu44Od44O844K644KS6Zmk5Y67XG4gICAgdGhpcy5wb3Nlcy5zaGlmdCgpO1xuXG4gICAgLy8g55S75YOP44Gu44Oe44O844K444Oz44KS5Y+W5b6XXG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGluZyBpbWFnZSBtYXJnaW5zLi4uYCk7XG4gICAgbGV0IGltYWdlVHJpbW1pbmc6XG4gICAgICB8IHtcbiAgICAgICAgICBtYXJnaW5Ub3A6IG51bWJlcjtcbiAgICAgICAgICBtYXJnaW5Cb3R0b206IG51bWJlcjtcbiAgICAgICAgICBoZWlnaHROZXc6IG51bWJlcjtcbiAgICAgICAgICBoZWlnaHRPbGQ6IG51bWJlcjtcbiAgICAgICAgICB3aWR0aDogbnVtYmVyO1xuICAgICAgICB9XG4gICAgICB8IHVuZGVmaW5lZCA9IHVuZGVmaW5lZDtcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICBjb25zdCBtYXJnaW5Db2xvciA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXRNYXJnaW5Db2xvcigpO1xuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RlZCBtYXJnaW4gY29sb3IuLi5gLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgbWFyZ2luQ29sb3JcbiAgICAgICk7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgPT09IG51bGwpIGNvbnRpbnVlO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yICE9PSB0aGlzLklNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUikge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIGNvbnN0IHRyaW1tZWQgPSBhd2FpdCBpbWFnZVRyaW1tZXIudHJpbU1hcmdpbihcbiAgICAgICAgbWFyZ2luQ29sb3IsXG4gICAgICAgIHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuICAgICAgaWYgKCF0cmltbWVkKSBjb250aW51ZTtcbiAgICAgIGltYWdlVHJpbW1pbmcgPSB0cmltbWVkO1xuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlcm1pbmVkIGltYWdlIHRyaW1taW5nIHBvc2l0aW9ucy4uLmAsXG4gICAgICAgIHRyaW1tZWRcbiAgICAgICk7XG4gICAgICBicmVhaztcbiAgICB9XG5cbiAgICAvLyDnlLvlg4/jgpLmlbTlvaJcbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCB8fCAhcG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIFByb2Nlc3NpbmcgaW1hZ2UuLi5gLFxuICAgICAgICBwb3NlLnRpbWVNaWxpc2Vjb25kc1xuICAgICAgKTtcblxuICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g44OV44Os44O844Og55S75YOPXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgaWYgKGltYWdlVHJpbW1pbmcpIHtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgICAgMCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLm1hcmdpblRvcCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLndpZHRoLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcuaGVpZ2h0TmV3XG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXBsYWNlQ29sb3IoXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX1NSQ19DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRFNUX0NPTE9SLFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9ESUZGX1RIUkVTSE9MRFxuICAgICAgKTtcblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlc2l6ZVdpdGhGaXQoe1xuICAgICAgICB3aWR0aDogdGhpcy5JTUFHRV9XSURUSCxcbiAgICAgIH0pO1xuXG4gICAgICBsZXQgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBmcmFtZSBpbWFnZWBcbiAgICAgICAgKTtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcblxuICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g44Od44O844K644OX44Os44OT44Ol44O855S75YOPXG4gICAgICBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLnBvc2VJbWFnZURhdGFVcmwpO1xuXG4gICAgICBpZiAoaW1hZ2VUcmltbWluZykge1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgICAwLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcubWFyZ2luVG9wLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcud2lkdGgsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5oZWlnaHROZXdcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlc2l6ZVdpdGhGaXQoe1xuICAgICAgICB3aWR0aDogdGhpcy5JTUFHRV9XSURUSCxcbiAgICAgIH0pO1xuXG4gICAgICBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICApO1xuICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIHBvc2UgcHJldmlldyBpbWFnZWBcbiAgICAgICAgKTtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuXG4gICAgICBpZiAocG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgLy8g55S75YOP44KS5pW05b2iIC0g6aGU44OV44Os44O844Og55S75YOPXG4gICAgICAgIGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICAgIG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgICApO1xuICAgICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIGZhY2UgZnJhbWUgaW1hZ2VgXG4gICAgICAgICAgKTtcbiAgICAgICAgICBjb250aW51ZTtcbiAgICAgICAgfVxuICAgICAgICBwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG4gICAgICB9XG4gICAgfVxuXG4gICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gIH1cblxuICByZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTogdm9pZCB7XG4gICAgLy8g5YWo44Od44O844K644KS5q+U6LyD44GX44Gm6aGe5Ly844Od44O844K644KS5YmK6ZmkXG4gICAgY29uc3QgbmV3UG9zZXM6IFBvc2VTZXRJdGVtW10gPSBbXTtcbiAgICBmb3IgKGNvbnN0IHBvc2VBIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpc0R1cGxpY2F0ZWQgPSBmYWxzZTtcbiAgICAgIGZvciAoY29uc3QgcG9zZUIgb2YgbmV3UG9zZXMpIHtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICAgIHBvc2VBLmJvZHlWZWN0b3JzLFxuICAgICAgICAgIHBvc2VCLmJvZHlWZWN0b3JzXG4gICAgICAgICk7XG4gICAgICAgIGNvbnN0IGlzU2ltaWxhckhhbmRQb3NlID1cbiAgICAgICAgICBwb3NlQS5oYW5kVmVjdG9ycyAmJiBwb3NlQi5oYW5kVmVjdG9yc1xuICAgICAgICAgICAgPyBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKHBvc2VBLmhhbmRWZWN0b3JzLCBwb3NlQi5oYW5kVmVjdG9ycylcbiAgICAgICAgICAgIDogZmFsc2U7XG5cbiAgICAgICAgaWYgKGlzU2ltaWxhckJvZHlQb3NlICYmIGlzU2ltaWxhckhhbmRQb3NlKSB7XG4gICAgICAgICAgLy8g6Lqr5L2T44O75omL44Go44KC44Gr6aGe5Ly844Od44O844K644Gq44KJ44GwXG4gICAgICAgICAgaXNEdXBsaWNhdGVkID0gdHJ1ZTtcbiAgICAgICAgICBicmVhaztcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBpZiAoaXNEdXBsaWNhdGVkKSBjb250aW51ZTtcblxuICAgICAgbmV3UG9zZXMucHVzaChwb3NlQSk7XG4gICAgfVxuXG4gICAgY29uc29sZS5pbmZvKFxuICAgICAgYFtQb3NlU2V0XSByZW1vdmVEdXBsaWNhdGVkUG9zZXMgLSBSZWR1Y2VkICR7dGhpcy5wb3Nlcy5sZW5ndGh9IHBvc2VzIC0+ICR7bmV3UG9zZXMubGVuZ3RofSBwb3Nlc2BcbiAgICApO1xuICAgIHRoaXMucG9zZXMgPSBuZXdQb3NlcztcbiAgfVxuXG4gIGdldFNpbWlsYXJQb3NlcyhcbiAgICByZXN1bHRzOiBSZXN1bHRzLFxuICAgIHRocmVzaG9sZDogbnVtYmVyID0gMC45XG4gICk6IFNpbWlsYXJQb3NlSXRlbVtdIHtcbiAgICBjb25zdCBib2R5VmVjdG9yID0gUG9zZVNldC5nZXRCb2R5VmVjdG9yKChyZXN1bHRzIGFzIGFueSkuZWEpO1xuICAgIGlmICghYm9keVZlY3RvcikgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yJztcblxuICAgIGNvbnN0IHBvc2VzID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICAgICAgcG9zZS5ib2R5VmVjdG9ycyxcbiAgICAgICAgYm9keVZlY3RvclxuICAgICAgKTtcbiAgICAgIGlmICh0aHJlc2hvbGQgPD0gc2ltaWxhcml0eSkge1xuICAgICAgICBwb3Nlcy5wdXNoKHtcbiAgICAgICAgICAuLi5wb3NlLFxuICAgICAgICAgIHNpbWlsYXJpdHk6IHNpbWlsYXJpdHksXG4gICAgICAgIH0pO1xuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBwb3NlcztcbiAgfVxuXG4gIHN0YXRpYyBnZXRCb2R5VmVjdG9yKFxuICAgIHBvc2VMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogQm9keVZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgc3RhdGljIGdldEhhbmRWZWN0b3JzKFxuICAgIGxlZnRIYW5kTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdLFxuICAgIHJpZ2h0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXVxuICApOiBIYW5kVmVjdG9yIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAoXG4gICAgICAocmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMCkgJiZcbiAgICAgIChsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMClcbiAgICApIHtcbiAgICAgIHJldHVybiB1bmRlZmluZWQ7XG4gICAgfVxuXG4gICAgcmV0dXJuIHtcbiAgICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICAgcmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbNF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1szXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbM10ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1syXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgcmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbOF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbN10ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1s2XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgICByaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTldLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnggLSByaWdodEhhbmRMYW5kbWFya3NbMThdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICAgbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1szXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1syXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgbGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnogLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTJdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTJdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTJdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzExXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTBdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g6Jas5oyHXG4gICAgICBsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTZdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTZdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTZdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE0XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE0XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE0XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICAgbGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1syMF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE4XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE4XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE4XS56LFxuICAgICAgICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgc3RhdGljIGlzU2ltaWxhckJvZHlQb3NlKFxuICAgIGJvZHlWZWN0b3JBOiBCb2R5VmVjdG9yLFxuICAgIGJvZHlWZWN0b3JCOiBCb2R5VmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuOFxuICApOiBib29sZWFuIHtcbiAgICBsZXQgaXNTaW1pbGFyID0gZmFsc2U7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KGJvZHlWZWN0b3JBLCBib2R5VmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkKSBpc1NpbWlsYXIgPSB0cnVlO1xuXG4gICAgLy8gY29uc29sZS5sb2coYFtQb3NlU2V0XSBpc1NpbWlsYXJQb3NlYCwgaXNTaW1pbGFyLCBzaW1pbGFyaXR5KTtcblxuICAgIHJldHVybiBpc1NpbWlsYXI7XG4gIH1cblxuICBzdGF0aWMgZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KFxuICAgIGJvZHlWZWN0b3JBOiBCb2R5VmVjdG9yLFxuICAgIGJvZHlWZWN0b3JCOiBCb2R5VmVjdG9yXG4gICk6IG51bWJlciB7XG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzID0ge1xuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLmxlZnRXcmlzdFRvTGVmdEVsYm93LFxuICAgICAgICBib2R5VmVjdG9yQi5sZWZ0V3Jpc3RUb0xlZnRFbGJvd1xuICAgICAgKSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0RWxib3dUb0xlZnRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXJcbiAgICAgICksXG4gICAgICByaWdodFdyaXN0VG9SaWdodEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5yaWdodFdyaXN0VG9SaWdodEVsYm93LFxuICAgICAgICBib2R5VmVjdG9yQi5yaWdodFdyaXN0VG9SaWdodEVsYm93XG4gICAgICApLFxuICAgICAgcmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcixcbiAgICAgICAgYm9keVZlY3RvckIucmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlclxuICAgICAgKSxcbiAgICB9O1xuXG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzU3VtID0gT2JqZWN0LnZhbHVlcyhjb3NTaW1pbGFyaXRpZXMpLnJlZHVjZShcbiAgICAgIChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSxcbiAgICAgIDBcbiAgICApO1xuICAgIHJldHVybiBjb3NTaW1pbGFyaXRpZXNTdW0gLyBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXMpLmxlbmd0aDtcbiAgfVxuXG4gIHN0YXRpYyBpc1NpbWlsYXJIYW5kUG9zZShcbiAgICBoYW5kVmVjdG9yQTogSGFuZFZlY3RvcixcbiAgICBoYW5kVmVjdG9yQjogSGFuZFZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjc1XG4gICk6IGJvb2xlYW4ge1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEhhbmRTaW1pbGFyaXR5KGhhbmRWZWN0b3JBLCBoYW5kVmVjdG9yQik7XG4gICAgaWYgKHNpbWlsYXJpdHkgPT09IC0xKSB7XG4gICAgICByZXR1cm4gdHJ1ZTtcbiAgICB9XG4gICAgcmV0dXJuIHNpbWlsYXJpdHkgPj0gdGhyZXNob2xkO1xuICB9XG5cbiAgc3RhdGljIGdldEhhbmRTaW1pbGFyaXR5KFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yXG4gICk6IG51bWJlciB7XG4gICAgY29uc3QgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kID1cbiAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsXG4gICAgICAgID8gdW5kZWZpbmVkXG4gICAgICAgIDoge1xuICAgICAgICAgICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgICAgICAgICByaWdodFRodW1iVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFRodW1iVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICAgICAgICAgcmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAgICAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g6Jas5oyHXG4gICAgICAgICAgICByaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgICAgICAgICByaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICB9O1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kID1cbiAgICAgIGhhbmRWZWN0b3JBLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsIHx8XG4gICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgICA/IHVuZGVmaW5lZFxuICAgICAgICA6IHtcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOimquaMh1xuICAgICAgICAgICAgbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgICAgICAgIGxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgICAgICAgICBsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICAgICAgICAgbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAgICAgICAgIGxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICBsZXQgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgPSAwO1xuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgPSBPYmplY3QudmFsdWVzKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZFxuICAgICAgKS5yZWR1Y2UoKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLCAwKTtcbiAgICB9XG5cbiAgICBsZXQgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kKSB7XG4gICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgPSBPYmplY3QudmFsdWVzKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCAmJiBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgcmV0dXJuIChcbiAgICAgICAgKGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCArIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kKSAvXG4gICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGggK1xuICAgICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoKVxuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCkge1xuICAgICAgcmV0dXJuIChcbiAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kIC9cbiAgICAgICAgT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH0gZWxzZSBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kIC9cbiAgICAgICAgT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQhKS5sZW5ndGhcbiAgICAgICk7XG4gICAgfSBlbHNlIHtcbiAgICAgIHJldHVybiAtMTtcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgYXN5bmMgZ2V0WmlwKCk6IFByb21pc2U8QmxvYj4ge1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAganNaaXAuZmlsZSgncG9zZXMuanNvbicsIGF3YWl0IHRoaXMuZ2V0SnNvbigpKTtcblxuICAgIGNvbnN0IGltYWdlRmlsZUV4dCA9IHRoaXMuZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZSh0aGlzLklNQUdFX01JTUUpO1xuXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLnBvc2VJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBwb3NlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZhY2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZmFjZSBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG5cbiAgICByZXR1cm4gYXdhaXQganNaaXAuZ2VuZXJhdGVBc3luYyh7IHR5cGU6ICdibG9iJyB9KTtcbiAgfVxuXG4gIGdldEZpbGVFeHRlbnNpb25CeU1pbWUoSU1BR0VfTUlNRTogc3RyaW5nKSB7XG4gICAgc3dpdGNoIChJTUFHRV9NSU1FKSB7XG4gICAgICBjYXNlICdpbWFnZS9wbmcnOlxuICAgICAgICByZXR1cm4gJ3BuZyc7XG4gICAgICBjYXNlICdpbWFnZS9qcGVnJzpcbiAgICAgICAgcmV0dXJuICdqcGcnO1xuICAgICAgY2FzZSAnaW1hZ2Uvd2VicCc6XG4gICAgICAgIHJldHVybiAnd2VicCc7XG4gICAgICBkZWZhdWx0OlxuICAgICAgICByZXR1cm4gJ3BuZyc7XG4gICAgfVxuICB9XG5cbiAgcHVibGljIGFzeW5jIGdldEpzb24oKTogUHJvbWlzZTxzdHJpbmc+IHtcbiAgICBpZiAodGhpcy52aWRlb01ldGFkYXRhID09PSB1bmRlZmluZWQgfHwgdGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKVxuICAgICAgcmV0dXJuICd7fSc7XG5cbiAgICBpZiAoIXRoaXMuaXNGaW5hbGl6ZWQpIHtcbiAgICAgIGF3YWl0IHRoaXMuZmluYWxpemUoKTtcbiAgICB9XG5cbiAgICBsZXQgcG9zZUxhbmRtYXJrTWFwcGluZ3MgPSBbXTtcbiAgICBmb3IgKGNvbnN0IGtleSBvZiBPYmplY3Qua2V5cyhQT1NFX0xBTkRNQVJLUykpIHtcbiAgICAgIGNvbnN0IGluZGV4OiBudW1iZXIgPSBQT1NFX0xBTkRNQVJLU1trZXkgYXMga2V5b2YgdHlwZW9mIFBPU0VfTEFORE1BUktTXTtcbiAgICAgIHBvc2VMYW5kbWFya01hcHBpbmdzW2luZGV4XSA9IGtleTtcbiAgICB9XG5cbiAgICBjb25zdCBqc29uOiBQb3NlU2V0SnNvbiA9IHtcbiAgICAgIGdlbmVyYXRvcjogJ21wLXZpZGVvLXBvc2UtZXh0cmFjdG9yJyxcbiAgICAgIHZlcnNpb246IDEsXG4gICAgICB2aWRlbzogdGhpcy52aWRlb01ldGFkYXRhISxcbiAgICAgIHBvc2VzOiB0aGlzLnBvc2VzLm1hcCgocG9zZTogUG9zZVNldEl0ZW0pOiBQb3NlU2V0SnNvbkl0ZW0gPT4ge1xuICAgICAgICAvLyBCb2R5VmVjdG9yIOOBruWcp+e4rlxuICAgICAgICBjb25zdCBib2R5VmVjdG9yID0gW107XG4gICAgICAgIGZvciAoY29uc3Qga2V5IG9mIFBvc2VTZXQuQk9EWV9WRUNUT1JfTUFQUElOR1MpIHtcbiAgICAgICAgICBib2R5VmVjdG9yLnB1c2gocG9zZS5ib2R5VmVjdG9yc1trZXkgYXMga2V5b2YgQm9keVZlY3Rvcl0pO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gSGFuZFZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgbGV0IGhhbmRWZWN0b3I6IChudW1iZXJbXSB8IG51bGwpW10gfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgICAgIGlmIChwb3NlLmhhbmRWZWN0b3JzKSB7XG4gICAgICAgICAgaGFuZFZlY3RvciA9IFtdO1xuICAgICAgICAgIGZvciAoY29uc3Qga2V5IG9mIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MpIHtcbiAgICAgICAgICAgIGhhbmRWZWN0b3IucHVzaChwb3NlLmhhbmRWZWN0b3JzW2tleSBhcyBrZXlvZiBIYW5kVmVjdG9yXSk7XG4gICAgICAgICAgfVxuICAgICAgICB9XG5cbiAgICAgICAgLy8gUG9zZVNldEpzb25JdGVtIOOBriBwb3NlIOOCquODluOCuOOCp+OCr+ODiOOCkueUn+aIkFxuICAgICAgICByZXR1cm4ge1xuICAgICAgICAgIHQ6IHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICAgIGQ6IHBvc2UuZHVyYXRpb25NaWxpc2Vjb25kcyxcbiAgICAgICAgICBwOiBwb3NlLnBvc2UsXG4gICAgICAgICAgbDogcG9zZS5sZWZ0SGFuZCxcbiAgICAgICAgICByOiBwb3NlLnJpZ2h0SGFuZCxcbiAgICAgICAgICB2OiBib2R5VmVjdG9yLFxuICAgICAgICAgIGg6IGhhbmRWZWN0b3IsXG4gICAgICAgICAgZTogcG9zZS5leHRlbmRlZERhdGEsXG4gICAgICAgIH07XG4gICAgICB9KSxcbiAgICAgIHBvc2VMYW5kbWFya01hcHBwaW5nczogcG9zZUxhbmRtYXJrTWFwcGluZ3MsXG4gICAgfTtcblxuICAgIHJldHVybiBKU09OLnN0cmluZ2lmeShqc29uKTtcbiAgfVxuXG4gIGxvYWRKc29uKGpzb246IHN0cmluZyB8IGFueSkge1xuICAgIGNvbnN0IHBhcnNlZEpzb24gPSB0eXBlb2YganNvbiA9PT0gJ3N0cmluZycgPyBKU09OLnBhcnNlKGpzb24pIDoganNvbjtcblxuICAgIGlmIChwYXJzZWRKc29uLmdlbmVyYXRvciAhPT0gJ21wLXZpZGVvLXBvc2UtZXh0cmFjdG9yJykge1xuICAgICAgdGhyb3cgJ+S4jeato+OBquODleOCoeOCpOODqyc7XG4gICAgfSBlbHNlIGlmIChwYXJzZWRKc29uLnZlcnNpb24gIT09IDEpIHtcbiAgICAgIHRocm93ICfmnKrlr77lv5zjga7jg5Djg7zjgrjjg6fjg7MnO1xuICAgIH1cblxuICAgIHRoaXMudmlkZW9NZXRhZGF0YSA9IHBhcnNlZEpzb24udmlkZW87XG4gICAgdGhpcy5wb3NlcyA9IHBhcnNlZEpzb24ucG9zZXMubWFwKChpdGVtOiBQb3NlU2V0SnNvbkl0ZW0pOiBQb3NlU2V0SXRlbSA9PiB7XG4gICAgICBjb25zdCBib2R5VmVjdG9yOiBhbnkgPSB7fTtcbiAgICAgIFBvc2VTZXQuQk9EWV9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgIGJvZHlWZWN0b3Jba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdID0gaXRlbS52W2luZGV4XTtcbiAgICAgIH0pO1xuXG4gICAgICBjb25zdCBoYW5kVmVjdG9yOiBhbnkgPSB7fTtcbiAgICAgIGlmIChpdGVtLmgpIHtcbiAgICAgICAgUG9zZVNldC5IQU5EX1ZFQ1RPUl9NQVBQSU5HUy5tYXAoKGtleSwgaW5kZXgpID0+IHtcbiAgICAgICAgICBoYW5kVmVjdG9yW2tleSBhcyBrZXlvZiBIYW5kVmVjdG9yXSA9IGl0ZW0uaCFbaW5kZXhdO1xuICAgICAgICB9KTtcbiAgICAgIH1cblxuICAgICAgcmV0dXJuIHtcbiAgICAgICAgdGltZU1pbGlzZWNvbmRzOiBpdGVtLnQsXG4gICAgICAgIGR1cmF0aW9uTWlsaXNlY29uZHM6IGl0ZW0uZCxcbiAgICAgICAgcG9zZTogaXRlbS5wLFxuICAgICAgICBsZWZ0SGFuZDogaXRlbS5sLFxuICAgICAgICByaWdodEhhbmQ6IGl0ZW0ucixcbiAgICAgICAgYm9keVZlY3RvcnM6IGJvZHlWZWN0b3IsXG4gICAgICAgIGhhbmRWZWN0b3JzOiBoYW5kVmVjdG9yLFxuICAgICAgICBmcmFtZUltYWdlRGF0YVVybDogdW5kZWZpbmVkLFxuICAgICAgICBleHRlbmRlZERhdGE6IGl0ZW0uZSxcbiAgICAgIH07XG4gICAgfSk7XG4gIH1cblxuICBhc3luYyBsb2FkWmlwKGJ1ZmZlcjogQXJyYXlCdWZmZXIsIGluY2x1ZGVJbWFnZXM6IGJvb2xlYW4gPSB0cnVlKSB7XG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBsb2FkWmlwLi4uYCwgSlNaaXApO1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBpbml0Li4uYCk7XG4gICAgY29uc3QgemlwID0gYXdhaXQganNaaXAubG9hZEFzeW5jKGJ1ZmZlciwgeyBiYXNlNjQ6IGZhbHNlIH0pO1xuICAgIGlmICghemlwKSB0aHJvdyAnWklQ44OV44Kh44Kk44Or44KS6Kqt44G/6L6844KB44G+44Gb44KT44Gn44GX44GfJztcblxuICAgIGNvbnN0IGpzb24gPSBhd2FpdCB6aXAuZmlsZSgncG9zZXMuanNvbicpPy5hc3luYygndGV4dCcpO1xuICAgIGlmIChqc29uID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgasgcG9zZS5qc29uIOOBjOWQq+OBvuOCjOOBpuOBhOOBvuOBm+OCkyc7XG4gICAgfVxuXG4gICAgdGhpcy5sb2FkSnNvbihqc29uKTtcblxuICAgIGNvbnN0IGZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGlmIChpbmNsdWRlSW1hZ2VzKSB7XG4gICAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBmcmFtZUltYWdlRmlsZU5hbWUgPSBgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKGZyYW1lSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IHBvc2VJbWFnZUZpbGVOYW1lID0gYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKHBvc2VJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cbiJdfQ==