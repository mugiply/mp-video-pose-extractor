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
        const handVector = PoseSet.gethandVector(results.leftHandLandmarks, results.rightHandLandmarks);
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
            bodyVector: bodyVector,
            handVector: handVector,
            frameImageDataUrl: frameImageDataUrl,
            poseImageDataUrl: poseImageDataUrl,
            faceFrameImageDataUrl: faceFrameImageDataUrl,
            extendedData: {},
        };
        if (1 <= this.poses.length) {
            // 前回のポーズとの類似性をチェック
            const lastPose = this.poses[this.poses.length - 1];
            const isSimilarBodyPose = PoseSet.isSimilarBodyPose(lastPose.bodyVector, pose.bodyVector);
            let isSimilarHandPose = true;
            if (lastPose.handVector && pose.handVector) {
                isSimilarHandPose = PoseSet.isSimilarHandPose(lastPose.handVector, pose.handVector);
            }
            else if (!lastPose.handVector && pose.handVector) {
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
                const isSimilarBodyPose = PoseSet.isSimilarBodyPose(poseA.bodyVector, poseB.bodyVector);
                const isSimilarHandPose = poseA.handVector && poseB.handVector
                    ? PoseSet.isSimilarHandPose(poseA.handVector, poseB.handVector)
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
    getSimilarPoses(results, threshold = 0.9, targetRange = 'all') {
        // 身体のベクトルを取得
        let bodyVector = PoseSet.getBodyVector(results.ea);
        if (!bodyVector) {
            throw 'Could not get the body vector';
        }
        // 手指のベクトルを取得
        let handVector;
        if (targetRange === 'all' || targetRange === 'handPose') {
            handVector = PoseSet.gethandVector(results.leftHandLandmarks, results.rightHandLandmarks);
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
                similarity = Math.max(bodySimilarity ?? 0, handSimilarity ?? 0);
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
            poses.push({
                ...pose,
                similarity: similarity,
                bodyPoseSimilarity: bodySimilarity,
                handPoseSimilarity: handSimilarity,
            });
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
    static gethandVector(leftHandLandmarks, rightHandLandmarks) {
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
                bodyVector: bodyVector,
                handVector: handVector,
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
];
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBd0VsQjtRQTlETyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQThDckMsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxxQkFBcUIsRUFBRSxDQUFDO1NBQ3pCLENBQUM7SUFDSixDQUFDO0lBRUQsWUFBWTtRQUNWLE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUM7SUFDakMsQ0FBQztJQUVELFlBQVksQ0FBQyxTQUFpQjtRQUM1QixJQUFJ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QUcsUUFBUSxJQUFJLENBQUMsVUFBVSxXQUFXLFdBQVcsRUFBRSxDQUFDO3FCQUN6RTtpQkFDRjthQUNGO1NBQ0Y7SUFDSCxDQUFDOztBQWpuQ0Qsa0JBQWtCO0FBQ0ssNEJBQW9CLEdBQUc7SUFDNUMsS0FBSztJQUNMLHdCQUF3QjtJQUN4QiwyQkFBMkI7SUFDM0IsS0FBSztJQUNMLHNCQUFzQjtJQUN0Qix5QkFBeUI7Q0FDMUIsQ0FBQztBQUVGLGtCQUFrQjtBQUNLLDRCQUFvQixHQUFHO0lBQzVDLFVBQVU7SUFDViwyQkFBMkI7SUFDM0IsbUNBQW1DO0lBQ25DLFlBQVk7SUFDWixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDVixrQ0FBa0M7SUFDbEMsMENBQTBDO0lBQzFDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwwQkFBMEI7SUFDMUIsa0NBQWtDO0lBQ2xDLFlBQVk7SUFDWixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwrQkFBK0I7SUFDL0IsdUNBQXVDO0lBQ3ZDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0NBQ3pDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBQT1NFX0xBTkRNQVJLUywgUmVzdWx0cyB9IGZyb20gJ0BtZWRpYXBpcGUvaG9saXN0aWMnO1xuaW1wb3J0ICogYXMgSlNaaXAgZnJvbSAnanN6aXAnO1xuaW1wb3J0IHsgUG9zZVNldEl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWl0ZW0nO1xuaW1wb3J0IHsgUG9zZVNldEpzb24gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24nO1xuaW1wb3J0IHsgUG9zZVNldEpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uLWl0ZW0nO1xuaW1wb3J0IHsgQm9keVZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvYm9keS12ZWN0b3InO1xuXG4vLyBAdHMtaWdub3JlXG5pbXBvcnQgY29zU2ltaWxhcml0eSBmcm9tICdjb3Mtc2ltaWxhcml0eSc7XG5pbXBvcnQgeyBTaW1pbGFyUG9zZUl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3NpbWlsYXItcG9zZS1pdGVtJztcbmltcG9ydCB7IEltYWdlVHJpbW1lciB9IGZyb20gJy4vaW50ZXJuYWxzL2ltYWdlLXRyaW1tZXInO1xuaW1wb3J0IHsgSGFuZFZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvaGFuZC12ZWN0b3InO1xuXG5leHBvcnQgY2xhc3MgUG9zZVNldCB7XG4gIHB1YmxpYyBnZW5lcmF0b3I/OiBzdHJpbmc7XG4gIHB1YmxpYyB2ZXJzaW9uPzogbnVtYmVyO1xuICBwcml2YXRlIHZpZGVvTWV0YWRhdGEhOiB7XG4gICAgbmFtZTogc3RyaW5nO1xuICAgIHdpZHRoOiBudW1iZXI7XG4gICAgaGVpZ2h0OiBudW1iZXI7XG4gICAgZHVyYXRpb246IG51bWJlcjtcbiAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IG51bWJlcjtcbiAgfTtcbiAgcHVibGljIHBvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gIHB1YmxpYyBpc0ZpbmFsaXplZD86IGJvb2xlYW4gPSBmYWxzZTtcblxuICAvLyBCb2R5VmVjdG9yIOOBruOCreODvOWQjVxuICBwdWJsaWMgc3RhdGljIHJlYWRvbmx5IEJPRFlfVkVDVE9SX01BUFBJTkdTID0gW1xuICAgIC8vIOWPs+iFlVxuICAgICdyaWdodFdyaXN0VG9SaWdodEVsYm93JyxcbiAgICAncmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcicsXG4gICAgLy8g5bem6IWVXG4gICAgJ2xlZnRXcmlzdFRvTGVmdEVsYm93JyxcbiAgICAnbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXInLFxuICBdO1xuXG4gIC8vIEhhbmRWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSEFORF9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgJ3JpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdyaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAncmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAncmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICdyaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAnbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICdsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICdsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gIF07XG5cbiAgLy8g55S75YOP5pu444GN5Ye644GX5pmC44Gu6Kit5a6aXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEg6IG51bWJlciA9IDEwODA7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUlNRTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPVxuICAgICdpbWFnZS93ZWJwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9RVUFMSVRZID0gMC44O1xuXG4gIC8vIOeUu+WDj+OBruS9meeZvemZpOWOu1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUiA9ICcjMDAwMDAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTEQgPSA1MDtcblxuICAvLyDnlLvlg4/jga7og4zmma/oibLnva7mj5tcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SID0gJyMwMTZBRkQnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IgPSAnI0ZGRkZGRjAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTEQgPSAxMzA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgZ2V0UG9zZXMoKTogUG9zZVNldEl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgZ2V0UG9zZUJ5VGltZSh0aW1lTWlsaXNlY29uZHM6IG51bWJlcik6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UudGltZU1pbGlzZWNvbmRzID09PSB0aW1lTWlsaXNlY29uZHMpO1xuICB9XG5cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHBvc2VJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICk6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHJldHVybjtcblxuICAgIGlmICh0aGlzLnBvc2VzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhpcy52aWRlb01ldGFkYXRhLmZpcnN0UG9zZURldGVjdGVkVGltZSA9IHZpZGVvVGltZU1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHBvc2Ugd2l0aCB0aGUgd29ybGQgY29vcmRpbmF0ZWAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgYm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3Rvcihwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZSk7XG4gICAgaWYgKCFib2R5VmVjdG9yKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yYCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGVcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgaWYgKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkICYmXG4gICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkXG4gICAgKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBsZWZ0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgcmlnaHQgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IGhhbmRWZWN0b3IgPSBQb3NlU2V0LmdldGhhbmRWZWN0b3IoXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICApO1xuICAgIGlmICghaGFuZFZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIHZlY3RvcmAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZTogUG9zZVNldEl0ZW0gPSB7XG4gICAgICB0aW1lTWlsaXNlY29uZHM6IHZpZGVvVGltZU1pbGlzZWNvbmRzLFxuICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogLTEsXG4gICAgICBwb3NlOiBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZS5tYXAoKHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueCxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay55LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnosXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsudmlzaWJpbGl0eSxcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgbGVmdEhhbmQ6IHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIHJpZ2h0SGFuZDogcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcz8ubWFwKChub3JtYWxpemVkTGFuZG1hcmspID0+IHtcbiAgICAgICAgcmV0dXJuIFtcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueCxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueSxcbiAgICAgICAgICBub3JtYWxpemVkTGFuZG1hcmsueixcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgYm9keVZlY3RvcjogYm9keVZlY3RvcixcbiAgICAgIGhhbmRWZWN0b3I6IGhhbmRWZWN0b3IsXG4gICAgICBmcmFtZUltYWdlRGF0YVVybDogZnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBwb3NlSW1hZ2VEYXRhVXJsOiBwb3NlSW1hZ2VEYXRhVXJsLFxuICAgICAgZmFjZUZyYW1lSW1hZ2VEYXRhVXJsOiBmYWNlRnJhbWVJbWFnZURhdGFVcmwsXG4gICAgICBleHRlbmRlZERhdGE6IHt9LFxuICAgIH07XG5cbiAgICBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgLy8g5YmN5Zue44Gu44Od44O844K644Go44Gu6aGe5Ly85oCn44KS44OB44Kn44OD44KvXG4gICAgICBjb25zdCBsYXN0UG9zZSA9IHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXTtcblxuICAgICAgY29uc3QgaXNTaW1pbGFyQm9keVBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckJvZHlQb3NlKFxuICAgICAgICBsYXN0UG9zZS5ib2R5VmVjdG9yLFxuICAgICAgICBwb3NlLmJvZHlWZWN0b3JcbiAgICAgICk7XG5cbiAgICAgIGxldCBpc1NpbWlsYXJIYW5kUG9zZSA9IHRydWU7XG4gICAgICBpZiAobGFzdFBvc2UuaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKFxuICAgICAgICAgIGxhc3RQb3NlLmhhbmRWZWN0b3IsXG4gICAgICAgICAgcG9zZS5oYW5kVmVjdG9yXG4gICAgICAgICk7XG4gICAgICB9IGVsc2UgaWYgKCFsYXN0UG9zZS5oYW5kVmVjdG9yICYmIHBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBpc1NpbWlsYXJIYW5kUG9zZSA9IGZhbHNlO1xuICAgICAgfVxuXG4gICAgICBpZiAoaXNTaW1pbGFyQm9keVBvc2UgJiYgaXNTaW1pbGFySGFuZFBvc2UpIHtcbiAgICAgICAgLy8g6Lqr5L2T44O75omL44Go44KC44Gr6aGe5Ly844Od44O844K644Gq44KJ44Gw44K544Kt44OD44OXXG4gICAgICAgIHJldHVybjtcbiAgICAgIH1cblxuICAgICAgLy8g5YmN5Zue44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgICBjb25zdCBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgIHZpZGVvVGltZU1pbGlzZWNvbmRzIC0gbGFzdFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcztcbiAgICB9XG5cbiAgICB0aGlzLnBvc2VzLnB1c2gocG9zZSk7XG5cbiAgICByZXR1cm4gcG9zZTtcbiAgfVxuXG4gIGFzeW5jIGZpbmFsaXplKCkge1xuICAgIGlmICgwID09IHRoaXMucG9zZXMubGVuZ3RoKSB7XG4gICAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgICAgIHJldHVybjtcbiAgICB9XG5cbiAgICAvLyDmnIDlvozjga7jg53jg7zjgrrjga7mjIHntprmmYLplpPjgpLoqK3lrppcbiAgICBpZiAoMSA8PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgY29uc3QgbGFzdFBvc2UgPSB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV07XG4gICAgICBpZiAobGFzdFBvc2UuZHVyYXRpb25NaWxpc2Vjb25kcyA9PSAtMSkge1xuICAgICAgICBjb25zdCBwb3NlRHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgICAgdGhpcy52aWRlb01ldGFkYXRhLmR1cmF0aW9uIC0gbGFzdFBvc2UudGltZU1pbGlzZWNvbmRzO1xuICAgICAgICB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV0uZHVyYXRpb25NaWxpc2Vjb25kcyA9XG4gICAgICAgICAgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgICB9XG4gICAgfVxuXG4gICAgLy8g6YeN6KSH44Od44O844K644KS6Zmk5Y67XG4gICAgdGhpcy5yZW1vdmVEdXBsaWNhdGVkUG9zZXMoKTtcblxuICAgIC8vIOacgOWIneOBruODneODvOOCuuOCkumZpOWOu1xuICAgIHRoaXMucG9zZXMuc2hpZnQoKTtcblxuICAgIC8vIOeUu+WDj+OBruODnuODvOOCuOODs+OCkuWPluW+l1xuICAgIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gZmluYWxpemUgLSBEZXRlY3RpbmcgaW1hZ2UgbWFyZ2lucy4uLmApO1xuICAgIGxldCBpbWFnZVRyaW1taW5nOlxuICAgICAgfCB7XG4gICAgICAgICAgbWFyZ2luVG9wOiBudW1iZXI7XG4gICAgICAgICAgbWFyZ2luQm90dG9tOiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0TmV3OiBudW1iZXI7XG4gICAgICAgICAgaGVpZ2h0T2xkOiBudW1iZXI7XG4gICAgICAgICAgd2lkdGg6IG51bWJlcjtcbiAgICAgICAgfVxuICAgICAgfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgY29uc3QgbWFyZ2luQ29sb3IgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0TWFyZ2luQ29sb3IoKTtcbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0ZWQgbWFyZ2luIGNvbG9yLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgIG1hcmdpbkNvbG9yXG4gICAgICApO1xuICAgICAgaWYgKG1hcmdpbkNvbG9yID09PSBudWxsKSBjb250aW51ZTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciAhPT0gdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfQ09MT1IpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG4gICAgICBjb25zdCB0cmltbWVkID0gYXdhaXQgaW1hZ2VUcmltbWVyLnRyaW1NYXJnaW4oXG4gICAgICAgIG1hcmdpbkNvbG9yLFxuICAgICAgICB0aGlzLklNQUdFX01BUkdJTl9UUklNTUlOR19ESUZGX1RIUkVTSE9MRFxuICAgICAgKTtcbiAgICAgIGlmICghdHJpbW1lZCkgY29udGludWU7XG4gICAgICBpbWFnZVRyaW1taW5nID0gdHJpbW1lZDtcbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZXJtaW5lZCBpbWFnZSB0cmltbWluZyBwb3NpdGlvbnMuLi5gLFxuICAgICAgICB0cmltbWVkXG4gICAgICApO1xuICAgICAgYnJlYWs7XG4gICAgfVxuXG4gICAgLy8g55S75YOP44KS5pW05b2iXG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGxldCBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwgfHwgIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgY29uc29sZS5sb2coXG4gICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBQcm9jZXNzaW5nIGltYWdlLi4uYCxcbiAgICAgICAgcG9zZS50aW1lTWlsaXNlY29uZHNcbiAgICAgICk7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODleODrOODvOODoOeUu+WDj1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVwbGFjZUNvbG9yKFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9TUkNfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RTVF9DT0xPUixcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbGV0IG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZnJhbWUgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOODneODvOOCuuODl+ODrOODk+ODpeODvOeUu+WDj1xuICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgaWYgKGltYWdlVHJpbW1pbmcpIHtcbiAgICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmNyb3AoXG4gICAgICAgICAgMCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLm1hcmdpblRvcCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLndpZHRoLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcuaGVpZ2h0TmV3XG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5yZXNpemVXaXRoRml0KHtcbiAgICAgICAgd2lkdGg6IHRoaXMuSU1BR0VfV0lEVEgsXG4gICAgICB9KTtcblxuICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICB0aGlzLklNQUdFX01JTUUsXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICA6IHVuZGVmaW5lZFxuICAgICAgKTtcbiAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBwb3NlIHByZXZpZXcgaW1hZ2VgXG4gICAgICAgICk7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcblxuICAgICAgaWYgKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIC8vIOeUu+WDj+OCkuaVtOW9oiAtIOmhlOODleODrOODvOODoOeUu+WDj1xuICAgICAgICBpbWFnZVRyaW1tZXIgPSBuZXcgSW1hZ2VUcmltbWVyKCk7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsKTtcblxuICAgICAgICBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICAgIHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL2pwZWcnIHx8IHRoaXMuSU1BR0VfTUlNRSA9PT0gJ2ltYWdlL3dlYnAnXG4gICAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICAgKTtcbiAgICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIENvdWxkIG5vdCBnZXQgdGhlIG5ldyBkYXRhdXJsIGZvciBmYWNlIGZyYW1lIGltYWdlYFxuICAgICAgICAgICk7XG4gICAgICAgICAgY29udGludWU7XG4gICAgICAgIH1cbiAgICAgICAgcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuICAgICAgfVxuICAgIH1cblxuICAgIHRoaXMuaXNGaW5hbGl6ZWQgPSB0cnVlO1xuICB9XG5cbiAgcmVtb3ZlRHVwbGljYXRlZFBvc2VzKCk6IHZvaWQge1xuICAgIC8vIOWFqOODneODvOOCuuOCkuavlOi8g+OBl+OBpumhnuS8vOODneODvOOCuuOCkuWJiumZpFxuICAgIGNvbnN0IG5ld1Bvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlQSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaXNEdXBsaWNhdGVkID0gZmFsc2U7XG4gICAgICBmb3IgKGNvbnN0IHBvc2VCIG9mIG5ld1Bvc2VzKSB7XG4gICAgICAgIGNvbnN0IGlzU2ltaWxhckJvZHlQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJCb2R5UG9zZShcbiAgICAgICAgICBwb3NlQS5ib2R5VmVjdG9yLFxuICAgICAgICAgIHBvc2VCLmJvZHlWZWN0b3JcbiAgICAgICAgKTtcbiAgICAgICAgY29uc3QgaXNTaW1pbGFySGFuZFBvc2UgPVxuICAgICAgICAgIHBvc2VBLmhhbmRWZWN0b3IgJiYgcG9zZUIuaGFuZFZlY3RvclxuICAgICAgICAgICAgPyBQb3NlU2V0LmlzU2ltaWxhckhhbmRQb3NlKHBvc2VBLmhhbmRWZWN0b3IsIHBvc2VCLmhhbmRWZWN0b3IpXG4gICAgICAgICAgICA6IGZhbHNlO1xuXG4gICAgICAgIGlmIChpc1NpbWlsYXJCb2R5UG9zZSAmJiBpc1NpbWlsYXJIYW5kUG9zZSkge1xuICAgICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBqOOCguOBq+mhnuS8vOODneODvOOCuuOBquOCieOBsFxuICAgICAgICAgIGlzRHVwbGljYXRlZCA9IHRydWU7XG4gICAgICAgICAgYnJlYWs7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKGlzRHVwbGljYXRlZCkgY29udGludWU7XG5cbiAgICAgIG5ld1Bvc2VzLnB1c2gocG9zZUEpO1xuICAgIH1cblxuICAgIGNvbnNvbGUuaW5mbyhcbiAgICAgIGBbUG9zZVNldF0gcmVtb3ZlRHVwbGljYXRlZFBvc2VzIC0gUmVkdWNlZCAke3RoaXMucG9zZXMubGVuZ3RofSBwb3NlcyAtPiAke25ld1Bvc2VzLmxlbmd0aH0gcG9zZXNgXG4gICAgKTtcbiAgICB0aGlzLnBvc2VzID0gbmV3UG9zZXM7XG4gIH1cblxuICBnZXRTaW1pbGFyUG9zZXMoXG4gICAgcmVzdWx0czogUmVzdWx0cyxcbiAgICB0aHJlc2hvbGQ6IG51bWJlciA9IDAuOSxcbiAgICB0YXJnZXRSYW5nZTogJ2FsbCcgfCAnYm9keVBvc2UnIHwgJ2hhbmRQb3NlJyA9ICdhbGwnXG4gICk6IFNpbWlsYXJQb3NlSXRlbVtdIHtcbiAgICAvLyDouqvkvZPjga7jg5njgq/jg4jjg6vjgpLlj5blvpdcbiAgICBsZXQgYm9keVZlY3RvcjogQm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3RvcigocmVzdWx0cyBhcyBhbnkpLmVhKTtcbiAgICBpZiAoIWJvZHlWZWN0b3IpIHtcbiAgICAgIHRocm93ICdDb3VsZCBub3QgZ2V0IHRoZSBib2R5IHZlY3Rvcic7XG4gICAgfVxuXG4gICAgLy8g5omL5oyH44Gu44OZ44Kv44OI44Or44KS5Y+W5b6XXG4gICAgbGV0IGhhbmRWZWN0b3I6IEhhbmRWZWN0b3I7XG4gICAgaWYgKHRhcmdldFJhbmdlID09PSAnYWxsJyB8fCB0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJykge1xuICAgICAgaGFuZFZlY3RvciA9IFBvc2VTZXQuZ2V0aGFuZFZlY3RvcihcbiAgICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyxcbiAgICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICAgICk7XG4gICAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIWhhbmRWZWN0b3IpIHtcbiAgICAgICAgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIGhhbmQgdmVjdG9yJztcbiAgICAgIH1cbiAgICB9XG5cbiAgICAvLyDlkITjg53jg7zjgrrjgajjg5njgq/jg4jjg6vjgpLmr5TovINcbiAgICBjb25zdCBwb3NlcyA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBpZiAoXG4gICAgICAgICh0YXJnZXRSYW5nZSA9PT0gJ2FsbCcgfHwgdGFyZ2V0UmFuZ2UgPT09ICdib2R5UG9zZScpICYmXG4gICAgICAgICFwb3NlLmJvZHlWZWN0b3JcbiAgICAgICkge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScgJiYgIXBvc2UuaGFuZFZlY3Rvcikge1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cblxuICAgICAgLy8g6Lqr5L2T44Gu44Od44O844K644Gu6aGe5Ly85bqm44KS5Y+W5b6XXG4gICAgICBsZXQgYm9keVNpbWlsYXJpdHk6IG51bWJlcjtcbiAgICAgIGlmIChib2R5VmVjdG9yICYmIHBvc2UuYm9keVZlY3Rvcikge1xuICAgICAgICBib2R5U2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0Qm9keVBvc2VTaW1pbGFyaXR5KFxuICAgICAgICAgIHBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgICBib2R5VmVjdG9yXG4gICAgICAgICk7XG4gICAgICB9XG5cbiAgICAgIC8vIOaJi+aMh+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGhhbmRTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaGFuZFNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEhhbmRTaW1pbGFyaXR5KHBvc2UuaGFuZFZlY3RvciwgaGFuZFZlY3Rvcik7XG4gICAgICB9XG5cbiAgICAgIC8vIOWIpOWumlxuICAgICAgbGV0IHNpbWlsYXJpdHk6IG51bWJlcixcbiAgICAgICAgaXNTaW1pbGFyID0gZmFsc2U7XG4gICAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdhbGwnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBNYXRoLm1heChib2R5U2ltaWxhcml0eSA/PyAwLCBoYW5kU2ltaWxhcml0eSA/PyAwKTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBib2R5U2ltaWxhcml0eSB8fCB0aHJlc2hvbGQgPD0gaGFuZFNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKHRhcmdldFJhbmdlID09PSAnYm9keVBvc2UnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBib2R5U2ltaWxhcml0eTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBib2R5U2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH0gZWxzZSBpZiAodGFyZ2V0UmFuZ2UgPT09ICdoYW5kUG9zZScpIHtcbiAgICAgICAgc2ltaWxhcml0eSA9IGhhbmRTaW1pbGFyaXR5O1xuICAgICAgICBpZiAodGhyZXNob2xkIDw9IGhhbmRTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfVxuXG4gICAgICBpZiAoIWlzU2ltaWxhcikgY29udGludWU7XG5cbiAgICAgIC8vIOe1kOaenOOBuOi/veWKoFxuICAgICAgcG9zZXMucHVzaCh7XG4gICAgICAgIC4uLnBvc2UsXG4gICAgICAgIHNpbWlsYXJpdHk6IHNpbWlsYXJpdHksXG4gICAgICAgIGJvZHlQb3NlU2ltaWxhcml0eTogYm9keVNpbWlsYXJpdHksXG4gICAgICAgIGhhbmRQb3NlU2ltaWxhcml0eTogaGFuZFNpbWlsYXJpdHksXG4gICAgICB9IGFzIFNpbWlsYXJQb3NlSXRlbSk7XG4gICAgfVxuXG4gICAgcmV0dXJuIHBvc2VzO1xuICB9XG5cbiAgc3RhdGljIGdldEJvZHlWZWN0b3IoXG4gICAgcG9zZUxhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXVxuICApOiBCb2R5VmVjdG9yIHwgdW5kZWZpbmVkIHtcbiAgICByZXR1cm4ge1xuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX1dSSVNUXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLlJJR0hUX0VMQk9XXS56LFxuICAgICAgXSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9TSE9VTERFUl0ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS56LFxuICAgICAgXSxcbiAgICAgIGxlZnRFbGJvd1RvTGVmdFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS54LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS56IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgIH07XG4gIH1cblxuICBzdGF0aWMgZ2V0aGFuZFZlY3RvcihcbiAgICBsZWZ0SGFuZExhbmRtYXJrczogeyB4OiBudW1iZXI7IHk6IG51bWJlcjsgejogbnVtYmVyIH1bXSxcbiAgICByaWdodEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogSGFuZFZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgaWYgKFxuICAgICAgKHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApICYmXG4gICAgICAobGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDApXG4gICAgKSB7XG4gICAgICByZXR1cm4gdW5kZWZpbmVkO1xuICAgIH1cblxuICAgIHJldHVybiB7XG4gICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnggLSByaWdodEhhbmRMYW5kbWFya3NbM10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzRdLnogLSByaWdodEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnggLSByaWdodEhhbmRMYW5kbWFya3NbN10ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbN10ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzhdLnogLSByaWdodEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnggLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnkgLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzddLnogLSByaWdodEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAgIHJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTldLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1szXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1syXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgIGxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDkuK3mjIdcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzEyXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTFdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMV0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzEwXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICAgbGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE2XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTVdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTVdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxNF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDlsI/mjIdcbiAgICAgIGxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMjBdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOV0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICBsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTldLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxOF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgfTtcbiAgfVxuXG4gIHN0YXRpYyBpc1NpbWlsYXJCb2R5UG9zZShcbiAgICBib2R5VmVjdG9yQTogQm9keVZlY3RvcixcbiAgICBib2R5VmVjdG9yQjogQm9keVZlY3RvcixcbiAgICB0aHJlc2hvbGQgPSAwLjhcbiAgKTogYm9vbGVhbiB7XG4gICAgbGV0IGlzU2ltaWxhciA9IGZhbHNlO1xuICAgIGNvbnN0IHNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShib2R5VmVjdG9yQSwgYm9keVZlY3RvckIpO1xuICAgIGlmIChzaW1pbGFyaXR5ID49IHRocmVzaG9sZCkgaXNTaW1pbGFyID0gdHJ1ZTtcblxuICAgIC8vIGNvbnNvbGUubG9nKGBbUG9zZVNldF0gaXNTaW1pbGFyUG9zZWAsIGlzU2ltaWxhciwgc2ltaWxhcml0eSk7XG5cbiAgICByZXR1cm4gaXNTaW1pbGFyO1xuICB9XG5cbiAgc3RhdGljIGdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICBib2R5VmVjdG9yQTogQm9keVZlY3RvcixcbiAgICBib2R5VmVjdG9yQjogQm9keVZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllcyA9IHtcbiAgICAgIGxlZnRXcmlzdFRvTGVmdEVsYm93OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5sZWZ0V3Jpc3RUb0xlZnRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIubGVmdFdyaXN0VG9MZWZ0RWxib3dcbiAgICAgICksXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEubGVmdEVsYm93VG9MZWZ0U2hvdWxkZXIsXG4gICAgICAgIGJvZHlWZWN0b3JCLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyXG4gICAgICApLFxuICAgICAgcmlnaHRXcmlzdFRvUmlnaHRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEucmlnaHRXcmlzdFRvUmlnaHRFbGJvdyxcbiAgICAgICAgYm9keVZlY3RvckIucmlnaHRXcmlzdFRvUmlnaHRFbGJvd1xuICAgICAgKSxcbiAgICAgIHJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXIsXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0RWxib3dUb1JpZ2h0U2hvdWxkZXJcbiAgICAgICksXG4gICAgfTtcblxuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1N1bSA9IE9iamVjdC52YWx1ZXMoY29zU2ltaWxhcml0aWVzKS5yZWR1Y2UoXG4gICAgICAoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsXG4gICAgICAwXG4gICAgKTtcbiAgICByZXR1cm4gY29zU2ltaWxhcml0aWVzU3VtIC8gT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzKS5sZW5ndGg7XG4gIH1cblxuICBzdGF0aWMgaXNTaW1pbGFySGFuZFBvc2UoXG4gICAgaGFuZFZlY3RvckE6IEhhbmRWZWN0b3IsXG4gICAgaGFuZFZlY3RvckI6IEhhbmRWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC43NVxuICApOiBib29sZWFuIHtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShoYW5kVmVjdG9yQSwgaGFuZFZlY3RvckIpO1xuICAgIGlmIChzaW1pbGFyaXR5ID09PSAtMSkge1xuICAgICAgcmV0dXJuIHRydWU7XG4gICAgfVxuICAgIHJldHVybiBzaW1pbGFyaXR5ID49IHRocmVzaG9sZDtcbiAgfVxuXG4gIHN0YXRpYyBnZXRIYW5kU2ltaWxhcml0eShcbiAgICBoYW5kVmVjdG9yQTogSGFuZFZlY3RvcixcbiAgICBoYW5kVmVjdG9yQjogSGFuZFZlY3RvclxuICApOiBudW1iZXIge1xuICAgIGNvbnN0IGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGwgfHxcbiAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbFxuICAgICAgICA/IHVuZGVmaW5lZFxuICAgICAgICA6IHtcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOimquaMh1xuICAgICAgICAgICAgcmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYlRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDkurrlt67jgZfmjIdcbiAgICAgICAgICAgIHJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgICAgICAgICByaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOiWrOaMh1xuICAgICAgICAgICAgcmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICAgICAgICAgcmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgfTtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICB9O1xuXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzTGVmdEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCA9IDA7XG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kXG4gICAgICApLnJlZHVjZSgoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsIDApO1xuICAgIH1cblxuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgJiYgY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIChjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgKyBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCkgL1xuICAgICAgICAoT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoICtcbiAgICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aClcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCEpLmxlbmd0aFxuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kKSB7XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH0gZWxzZSB7XG4gICAgICByZXR1cm4gLTE7XG4gICAgfVxuICB9XG5cbiAgcHVibGljIGFzeW5jIGdldFppcCgpOiBQcm9taXNlPEJsb2I+IHtcbiAgICBjb25zdCBqc1ppcCA9IG5ldyBKU1ppcCgpO1xuICAgIGpzWmlwLmZpbGUoJ3Bvc2VzLmpzb24nLCBhd2FpdCB0aGlzLmdldEpzb24oKSk7XG5cbiAgICBjb25zdCBpbWFnZUZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBpZiAocG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwuaW5kZXhPZignYmFzZTY0LCcpICsgJ2Jhc2U2NCwnLmxlbmd0aDtcbiAgICAgICAgICBjb25zdCBiYXNlNjQgPSBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAocG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIHRyeSB7XG4gICAgICAgICAgY29uc3QgaW5kZXggPVxuICAgICAgICAgICAgcG9zZS5wb3NlSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5wb3NlSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgcG9zZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmcmFtZSBpbWFnZWAsXG4gICAgICAgICAgICBlcnJvclxuICAgICAgICAgICk7XG4gICAgICAgICAgdGhyb3cgZXJyb3I7XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwuc3Vic3RyaW5nKGluZGV4KTtcbiAgICAgICAgICBqc1ppcC5maWxlKGBmYWNlLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZhY2UgZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgfVxuXG4gICAgcmV0dXJuIGF3YWl0IGpzWmlwLmdlbmVyYXRlQXN5bmMoeyB0eXBlOiAnYmxvYicgfSk7XG4gIH1cblxuICBnZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKElNQUdFX01JTUU6IHN0cmluZykge1xuICAgIHN3aXRjaCAoSU1BR0VfTUlNRSkge1xuICAgICAgY2FzZSAnaW1hZ2UvcG5nJzpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgICAgY2FzZSAnaW1hZ2UvanBlZyc6XG4gICAgICAgIHJldHVybiAnanBnJztcbiAgICAgIGNhc2UgJ2ltYWdlL3dlYnAnOlxuICAgICAgICByZXR1cm4gJ3dlYnAnO1xuICAgICAgZGVmYXVsdDpcbiAgICAgICAgcmV0dXJuICdwbmcnO1xuICAgIH1cbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRKc29uKCk6IFByb21pc2U8c3RyaW5nPiB7XG4gICAgaWYgKHRoaXMudmlkZW9NZXRhZGF0YSA9PT0gdW5kZWZpbmVkIHx8IHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZClcbiAgICAgIHJldHVybiAne30nO1xuXG4gICAgaWYgKCF0aGlzLmlzRmluYWxpemVkKSB7XG4gICAgICBhd2FpdCB0aGlzLmZpbmFsaXplKCk7XG4gICAgfVxuXG4gICAgbGV0IHBvc2VMYW5kbWFya01hcHBpbmdzID0gW107XG4gICAgZm9yIChjb25zdCBrZXkgb2YgT2JqZWN0LmtleXMoUE9TRV9MQU5ETUFSS1MpKSB7XG4gICAgICBjb25zdCBpbmRleDogbnVtYmVyID0gUE9TRV9MQU5ETUFSS1Nba2V5IGFzIGtleW9mIHR5cGVvZiBQT1NFX0xBTkRNQVJLU107XG4gICAgICBwb3NlTGFuZG1hcmtNYXBwaW5nc1tpbmRleF0gPSBrZXk7XG4gICAgfVxuXG4gICAgY29uc3QganNvbjogUG9zZVNldEpzb24gPSB7XG4gICAgICBnZW5lcmF0b3I6ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicsXG4gICAgICB2ZXJzaW9uOiAxLFxuICAgICAgdmlkZW86IHRoaXMudmlkZW9NZXRhZGF0YSEsXG4gICAgICBwb3NlczogdGhpcy5wb3Nlcy5tYXAoKHBvc2U6IFBvc2VTZXRJdGVtKTogUG9zZVNldEpzb25JdGVtID0+IHtcbiAgICAgICAgLy8gQm9keVZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgY29uc3QgYm9keVZlY3RvciA9IFtdO1xuICAgICAgICBmb3IgKGNvbnN0IGtleSBvZiBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTKSB7XG4gICAgICAgICAgYm9keVZlY3Rvci5wdXNoKHBvc2UuYm9keVZlY3RvcltrZXkgYXMga2V5b2YgQm9keVZlY3Rvcl0pO1xuICAgICAgICB9XG5cbiAgICAgICAgLy8gSGFuZFZlY3RvciDjga7lnKfnuK5cbiAgICAgICAgbGV0IGhhbmRWZWN0b3I6IChudW1iZXJbXSB8IG51bGwpW10gfCB1bmRlZmluZWQgPSB1bmRlZmluZWQ7XG4gICAgICAgIGlmIChwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgICBoYW5kVmVjdG9yID0gW107XG4gICAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZVNldC5IQU5EX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgICAgaGFuZFZlY3Rvci5wdXNoKHBvc2UuaGFuZFZlY3RvcltrZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0pO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuXG4gICAgICAgIC8vIFBvc2VTZXRKc29uSXRlbSDjga4gcG9zZSDjgqrjg5bjgrjjgqfjgq/jg4jjgpLnlJ/miJBcbiAgICAgICAgcmV0dXJuIHtcbiAgICAgICAgICB0OiBwb3NlLnRpbWVNaWxpc2Vjb25kcyxcbiAgICAgICAgICBkOiBwb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMsXG4gICAgICAgICAgcDogcG9zZS5wb3NlLFxuICAgICAgICAgIGw6IHBvc2UubGVmdEhhbmQsXG4gICAgICAgICAgcjogcG9zZS5yaWdodEhhbmQsXG4gICAgICAgICAgdjogYm9keVZlY3RvcixcbiAgICAgICAgICBoOiBoYW5kVmVjdG9yLFxuICAgICAgICAgIGU6IHBvc2UuZXh0ZW5kZWREYXRhLFxuICAgICAgICB9O1xuICAgICAgfSksXG4gICAgICBwb3NlTGFuZG1hcmtNYXBwcGluZ3M6IHBvc2VMYW5kbWFya01hcHBpbmdzLFxuICAgIH07XG5cbiAgICByZXR1cm4gSlNPTi5zdHJpbmdpZnkoanNvbik7XG4gIH1cblxuICBsb2FkSnNvbihqc29uOiBzdHJpbmcgfCBhbnkpIHtcbiAgICBjb25zdCBwYXJzZWRKc29uID0gdHlwZW9mIGpzb24gPT09ICdzdHJpbmcnID8gSlNPTi5wYXJzZShqc29uKSA6IGpzb247XG5cbiAgICBpZiAocGFyc2VkSnNvbi5nZW5lcmF0b3IgIT09ICdtcC12aWRlby1wb3NlLWV4dHJhY3RvcicpIHtcbiAgICAgIHRocm93ICfkuI3mraPjgarjg5XjgqHjgqTjg6snO1xuICAgIH0gZWxzZSBpZiAocGFyc2VkSnNvbi52ZXJzaW9uICE9PSAxKSB7XG4gICAgICB0aHJvdyAn5pyq5a++5b+c44Gu44OQ44O844K444On44OzJztcbiAgICB9XG5cbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEgPSBwYXJzZWRKc29uLnZpZGVvO1xuICAgIHRoaXMucG9zZXMgPSBwYXJzZWRKc29uLnBvc2VzLm1hcCgoaXRlbTogUG9zZVNldEpzb25JdGVtKTogUG9zZVNldEl0ZW0gPT4ge1xuICAgICAgY29uc3QgYm9keVZlY3RvcjogYW55ID0ge307XG4gICAgICBQb3NlU2V0LkJPRFlfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICBib2R5VmVjdG9yW2tleSBhcyBrZXlvZiBCb2R5VmVjdG9yXSA9IGl0ZW0udltpbmRleF07XG4gICAgICB9KTtcblxuICAgICAgY29uc3QgaGFuZFZlY3RvcjogYW55ID0ge307XG4gICAgICBpZiAoaXRlbS5oKSB7XG4gICAgICAgIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MubWFwKChrZXksIGluZGV4KSA9PiB7XG4gICAgICAgICAgaGFuZFZlY3RvcltrZXkgYXMga2V5b2YgSGFuZFZlY3Rvcl0gPSBpdGVtLmghW2luZGV4XTtcbiAgICAgICAgfSk7XG4gICAgICB9XG5cbiAgICAgIHJldHVybiB7XG4gICAgICAgIHRpbWVNaWxpc2Vjb25kczogaXRlbS50LFxuICAgICAgICBkdXJhdGlvbk1pbGlzZWNvbmRzOiBpdGVtLmQsXG4gICAgICAgIHBvc2U6IGl0ZW0ucCxcbiAgICAgICAgbGVmdEhhbmQ6IGl0ZW0ubCxcbiAgICAgICAgcmlnaHRIYW5kOiBpdGVtLnIsXG4gICAgICAgIGJvZHlWZWN0b3I6IGJvZHlWZWN0b3IsXG4gICAgICAgIGhhbmRWZWN0b3I6IGhhbmRWZWN0b3IsXG4gICAgICAgIGZyYW1lSW1hZ2VEYXRhVXJsOiB1bmRlZmluZWQsXG4gICAgICAgIGV4dGVuZGVkRGF0YTogaXRlbS5lLFxuICAgICAgfTtcbiAgICB9KTtcbiAgfVxuXG4gIGFzeW5jIGxvYWRaaXAoYnVmZmVyOiBBcnJheUJ1ZmZlciwgaW5jbHVkZUltYWdlczogYm9vbGVhbiA9IHRydWUpIHtcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGxvYWRaaXAuLi5gLCBKU1ppcCk7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGluaXQuLi5gKTtcbiAgICBjb25zdCB6aXAgPSBhd2FpdCBqc1ppcC5sb2FkQXN5bmMoYnVmZmVyLCB7IGJhc2U2NDogZmFsc2UgfSk7XG4gICAgaWYgKCF6aXApIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgpLoqq3jgb/ovrzjgoHjgb7jgZvjgpPjgafjgZfjgZ8nO1xuXG4gICAgY29uc3QganNvbiA9IGF3YWl0IHppcC5maWxlKCdwb3Nlcy5qc29uJyk/LmFzeW5jKCd0ZXh0Jyk7XG4gICAgaWYgKGpzb24gPT09IHVuZGVmaW5lZCkge1xuICAgICAgdGhyb3cgJ1pJUOODleOCoeOCpOODq+OBqyBwb3NlLmpzb24g44GM5ZCr44G+44KM44Gm44GE44G+44Gb44KTJztcbiAgICB9XG5cbiAgICB0aGlzLmxvYWRKc29uKGpzb24pO1xuXG4gICAgY29uc3QgZmlsZUV4dCA9IHRoaXMuZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZSh0aGlzLklNQUdFX01JTUUpO1xuXG4gICAgaWYgKGluY2x1ZGVJbWFnZXMpIHtcbiAgICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICAgIGlmICghcG9zZS5mcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IGZyYW1lSW1hZ2VGaWxlTmFtZSA9IGBmcmFtZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ZpbGVFeHR9YDtcbiAgICAgICAgICBjb25zdCBpbWFnZUJhc2U2NCA9IGF3YWl0IHppcFxuICAgICAgICAgICAgLmZpbGUoZnJhbWVJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsID0gYGRhdGE6JHt0aGlzLklNQUdFX01JTUV9O2Jhc2U2NCwke2ltYWdlQmFzZTY0fWA7XG4gICAgICAgICAgfVxuICAgICAgICB9XG4gICAgICAgIGlmICghcG9zZS5wb3NlSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgICAgY29uc3QgcG9zZUltYWdlRmlsZU5hbWUgPSBgcG9zZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ZpbGVFeHR9YDtcbiAgICAgICAgICBjb25zdCBpbWFnZUJhc2U2NCA9IGF3YWl0IHppcFxuICAgICAgICAgICAgLmZpbGUocG9zZUltYWdlRmlsZU5hbWUpXG4gICAgICAgICAgICA/LmFzeW5jKCdiYXNlNjQnKTtcbiAgICAgICAgICBpZiAoaW1hZ2VCYXNlNjQpIHtcbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cbiAgfVxufVxuIl19