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
            leftHand: results.leftHandLandmarks?.map((normalizedLandmark) => {
                return [
                    normalizedLandmark.x,
                    normalizedLandmark.y,
                    normalizedLandmark.z,
                ];
            }),
            rightHand: results.rightHandLandmarks?.map((normalizedLandmark) => {
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
            /*console.log(
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
                console.log(`[PoseSet] getHandSimilarity - Adjust similarity, because left hand not found...`);
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
                console.log(`[PoseSet] getHandSimilarity - Adjust similarity, because right hand not found...`);
                return (cosSimilaritiesSumLeftHand /
                    (Object.keys(cosSimilaritiesLeftHand).length * 2));
            }
            return (cosSimilaritiesSumLeftHand /
                Object.keys(cosSimilaritiesLeftHand).length);
        }
        return -1;
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
//# sourceMappingURL=data:application/json;base64,eyJ2ZXJzaW9uIjozLCJmaWxlIjoicG9zZS1zZXQuanMiLCJzb3VyY2VSb290IjoiIiwic291cmNlcyI6WyIuLi8uLi8uLi8uLi8uLi9wcm9qZWN0cy9uZ3gtbXAtcG9zZS1leHRyYWN0b3Ivc3JjL2xpYi9jbGFzc2VzL3Bvc2Utc2V0LnRzIl0sIm5hbWVzIjpbXSwibWFwcGluZ3MiOiJBQUFBLE9BQU8sRUFBRSxjQUFjLEVBQVcsTUFBTSxxQkFBcUIsQ0FBQztBQUM5RCxPQUFPLEtBQUssS0FBSyxNQUFNLE9BQU8sQ0FBQztBQU0vQixhQUFhO0FBQ2IsT0FBTyxhQUFhLE1BQU0sZ0JBQWdCLENBQUM7QUFFM0MsT0FBTyxFQUFFLFlBQVksRUFBRSxNQUFNLDJCQUEyQixDQUFDO0FBR3pELE1BQU0sT0FBTyxPQUFPO0lBd0VsQjtRQTlETyxVQUFLLEdBQWtCLEVBQUUsQ0FBQztRQUMxQixnQkFBVyxHQUFhLEtBQUssQ0FBQztRQThDckMsYUFBYTtRQUNJLGdCQUFXLEdBQVcsSUFBSSxDQUFDO1FBQzNCLGVBQVUsR0FDekIsWUFBWSxDQUFDO1FBQ0Usa0JBQWEsR0FBRyxHQUFHLENBQUM7UUFFckMsVUFBVTtRQUNPLGdDQUEyQixHQUFHLFNBQVMsQ0FBQztRQUN4Qyx5Q0FBb0MsR0FBRyxFQUFFLENBQUM7UUFFM0QsV0FBVztRQUNNLHVDQUFrQyxHQUFHLFNBQVMsQ0FBQztRQUMvQyx1Q0FBa0MsR0FBRyxXQUFXLENBQUM7UUFDakQsNENBQXVDLEdBQUcsR0FBRyxDQUFDO1FBRzdELElBQUksQ0FBQyxhQUFhLEdBQUc7WUFDbkIsSUFBSSxFQUFFLEVBQUU7WUFDUixLQUFLLEVBQUUsQ0FBQztZQUNSLE1BQU0sRUFBRSxDQUFDO1lBQ1QsUUFBUSxFQUFFLENBQUM7WUFDWCxxQkFBcUIsRUFBRSxDQUFDO1NBQ3pCLENBQUM7SUFDSixDQUFDO0lBRUQsWUFBWTtRQUNWLE9BQU8sSUFBSSxDQUFDLGFBQWEsQ0FBQyxJQUFJLENBQUM7SUFDakMsQ0FBQztJQUVELFlBQVksQ0FBQyxTQUFpQjtRQUM1QixJQUFJ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MkI7SUFDM0IsbUNBQW1DO0lBQ25DLFlBQVk7SUFDWixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDVixrQ0FBa0M7SUFDbEMsMENBQTBDO0lBQzFDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwwQkFBMEI7SUFDMUIsa0NBQWtDO0lBQ2xDLFlBQVk7SUFDWixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0lBQ3hDLFVBQVU7SUFDVixpQ0FBaUM7SUFDakMseUNBQXlDO0lBQ3pDLFVBQVU7SUFDViwrQkFBK0I7SUFDL0IsdUNBQXVDO0lBQ3ZDLFVBQVU7SUFDVixnQ0FBZ0M7SUFDaEMsd0NBQXdDO0NBQ3pDLENBQUMiLCJzb3VyY2VzQ29udGVudCI6WyJpbXBvcnQgeyBQT1NFX0xBTkRNQVJLUywgUmVzdWx0cyB9IGZyb20gJ0BtZWRpYXBpcGUvaG9saXN0aWMnO1xuaW1wb3J0ICogYXMgSlNaaXAgZnJvbSAnanN6aXAnO1xuaW1wb3J0IHsgUG9zZVNldEl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWl0ZW0nO1xuaW1wb3J0IHsgUG9zZVNldEpzb24gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3Bvc2Utc2V0LWpzb24nO1xuaW1wb3J0IHsgUG9zZVNldEpzb25JdGVtIH0gZnJvbSAnLi4vaW50ZXJmYWNlcy9wb3NlLXNldC1qc29uLWl0ZW0nO1xuaW1wb3J0IHsgQm9keVZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvYm9keS12ZWN0b3InO1xuXG4vLyBAdHMtaWdub3JlXG5pbXBvcnQgY29zU2ltaWxhcml0eSBmcm9tICdjb3Mtc2ltaWxhcml0eSc7XG5pbXBvcnQgeyBTaW1pbGFyUG9zZUl0ZW0gfSBmcm9tICcuLi9pbnRlcmZhY2VzL3NpbWlsYXItcG9zZS1pdGVtJztcbmltcG9ydCB7IEltYWdlVHJpbW1lciB9IGZyb20gJy4vaW50ZXJuYWxzL2ltYWdlLXRyaW1tZXInO1xuaW1wb3J0IHsgSGFuZFZlY3RvciB9IGZyb20gJy4uL2ludGVyZmFjZXMvaGFuZC12ZWN0b3InO1xuXG5leHBvcnQgY2xhc3MgUG9zZVNldCB7XG4gIHB1YmxpYyBnZW5lcmF0b3I/OiBzdHJpbmc7XG4gIHB1YmxpYyB2ZXJzaW9uPzogbnVtYmVyO1xuICBwcml2YXRlIHZpZGVvTWV0YWRhdGEhOiB7XG4gICAgbmFtZTogc3RyaW5nO1xuICAgIHdpZHRoOiBudW1iZXI7XG4gICAgaGVpZ2h0OiBudW1iZXI7XG4gICAgZHVyYXRpb246IG51bWJlcjtcbiAgICBmaXJzdFBvc2VEZXRlY3RlZFRpbWU6IG51bWJlcjtcbiAgfTtcbiAgcHVibGljIHBvc2VzOiBQb3NlU2V0SXRlbVtdID0gW107XG4gIHB1YmxpYyBpc0ZpbmFsaXplZD86IGJvb2xlYW4gPSBmYWxzZTtcblxuICAvLyBCb2R5VmVjdG9yIOOBruOCreODvOWQjVxuICBwdWJsaWMgc3RhdGljIHJlYWRvbmx5IEJPRFlfVkVDVE9SX01BUFBJTkdTID0gW1xuICAgIC8vIOWPs+iFlVxuICAgICdyaWdodFdyaXN0VG9SaWdodEVsYm93JyxcbiAgICAncmlnaHRFbGJvd1RvUmlnaHRTaG91bGRlcicsXG4gICAgLy8g5bem6IWVXG4gICAgJ2xlZnRXcmlzdFRvTGVmdEVsYm93JyxcbiAgICAnbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXInLFxuICBdO1xuXG4gIC8vIEhhbmRWZWN0b3Ig44Gu44Kt44O85ZCNXG4gIHB1YmxpYyBzdGF0aWMgcmVhZG9ubHkgSEFORF9WRUNUT1JfTUFQUElOR1MgPSBbXG4gICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgJ3JpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdyaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDkuK3mjIdcbiAgICAncmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdyaWdodE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAncmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOWPs+aJiyAtIOWwj+aMh1xuICAgICdyaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50JyxcbiAgICAncmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50JyxcbiAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAnbGVmdFRodW1iVGlwVG9GaXJzdEpvaW50JyxcbiAgICAnbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOS6uuW3ruOBl+aMh1xuICAgICdsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgJ2xlZnRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOiWrOaMh1xuICAgICdsZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCcsXG4gICAgJ2xlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQnLFxuICAgIC8vIOW3puaJiyAtIOWwj+aMh1xuICAgICdsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQnLFxuICAgICdsZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCcsXG4gIF07XG5cbiAgLy8g55S75YOP5pu444GN5Ye644GX5pmC44Gu6Kit5a6aXG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfV0lEVEg6IG51bWJlciA9IDEwODA7XG4gIHByaXZhdGUgcmVhZG9ubHkgSU1BR0VfTUlNRTogJ2ltYWdlL2pwZWcnIHwgJ2ltYWdlL3BuZycgfCAnaW1hZ2Uvd2VicCcgPVxuICAgICdpbWFnZS93ZWJwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9RVUFMSVRZID0gMC44O1xuXG4gIC8vIOeUu+WDj+OBruS9meeZvemZpOWOu1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX01BUkdJTl9UUklNTUlOR19DT0xPUiA9ICcjMDAwMDAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTEQgPSA1MDtcblxuICAvLyDnlLvlg4/jga7og4zmma/oibLnva7mj5tcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SID0gJyMwMTZBRkQnO1xuICBwcml2YXRlIHJlYWRvbmx5IElNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IgPSAnI0ZGRkZGRjAwJztcbiAgcHJpdmF0ZSByZWFkb25seSBJTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfRElGRl9USFJFU0hPTEQgPSAxMzA7XG5cbiAgY29uc3RydWN0b3IoKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0ge1xuICAgICAgbmFtZTogJycsXG4gICAgICB3aWR0aDogMCxcbiAgICAgIGhlaWdodDogMCxcbiAgICAgIGR1cmF0aW9uOiAwLFxuICAgICAgZmlyc3RQb3NlRGV0ZWN0ZWRUaW1lOiAwLFxuICAgIH07XG4gIH1cblxuICBnZXRWaWRlb05hbWUoKSB7XG4gICAgcmV0dXJuIHRoaXMudmlkZW9NZXRhZGF0YS5uYW1lO1xuICB9XG5cbiAgc2V0VmlkZW9OYW1lKHZpZGVvTmFtZTogc3RyaW5nKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLm5hbWUgPSB2aWRlb05hbWU7XG4gIH1cblxuICBzZXRWaWRlb01ldGFEYXRhKHdpZHRoOiBudW1iZXIsIGhlaWdodDogbnVtYmVyLCBkdXJhdGlvbjogbnVtYmVyKSB7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLndpZHRoID0gd2lkdGg7XG4gICAgdGhpcy52aWRlb01ldGFkYXRhLmhlaWdodCA9IGhlaWdodDtcbiAgICB0aGlzLnZpZGVvTWV0YWRhdGEuZHVyYXRpb24gPSBkdXJhdGlvbjtcbiAgfVxuXG4gIGdldE51bWJlck9mUG9zZXMoKTogbnVtYmVyIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gLTE7XG4gICAgcmV0dXJuIHRoaXMucG9zZXMubGVuZ3RoO1xuICB9XG5cbiAgZ2V0UG9zZXMoKTogUG9zZVNldEl0ZW1bXSB7XG4gICAgaWYgKHRoaXMucG9zZXMgPT09IHVuZGVmaW5lZCkgcmV0dXJuIFtdO1xuICAgIHJldHVybiB0aGlzLnBvc2VzO1xuICB9XG5cbiAgZ2V0UG9zZUJ5VGltZSh0aW1lTWlsaXNlY29uZHM6IG51bWJlcik6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAodGhpcy5wb3NlcyA9PT0gdW5kZWZpbmVkKSByZXR1cm4gdW5kZWZpbmVkO1xuICAgIHJldHVybiB0aGlzLnBvc2VzLmZpbmQoKHBvc2UpID0+IHBvc2UudGltZU1pbGlzZWNvbmRzID09PSB0aW1lTWlsaXNlY29uZHMpO1xuICB9XG5cbiAgcHVzaFBvc2UoXG4gICAgdmlkZW9UaW1lTWlsaXNlY29uZHM6IG51bWJlcixcbiAgICBmcmFtZUltYWdlRGF0YVVybDogc3RyaW5nIHwgdW5kZWZpbmVkLFxuICAgIHBvc2VJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICBmYWNlRnJhbWVJbWFnZURhdGFVcmw6IHN0cmluZyB8IHVuZGVmaW5lZCxcbiAgICByZXN1bHRzOiBSZXN1bHRzXG4gICk6IFBvc2VTZXRJdGVtIHwgdW5kZWZpbmVkIHtcbiAgICBpZiAocmVzdWx0cy5wb3NlTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHJldHVybjtcblxuICAgIGlmICh0aGlzLnBvc2VzLmxlbmd0aCA9PT0gMCkge1xuICAgICAgdGhpcy52aWRlb01ldGFkYXRhLmZpcnN0UG9zZURldGVjdGVkVGltZSA9IHZpZGVvVGltZU1pbGlzZWNvbmRzO1xuICAgIH1cblxuICAgIGNvbnN0IHBvc2VMYW5kbWFya3NXaXRoV29ybGRDb29yZGluYXRlOiBhbnlbXSA9IChyZXN1bHRzIGFzIGFueSkuZWFcbiAgICAgID8gKHJlc3VsdHMgYXMgYW55KS5lYVxuICAgICAgOiBbXTtcbiAgICBpZiAocG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGUubGVuZ3RoID09PSAwKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIHBvc2Ugd2l0aCB0aGUgd29ybGQgY29vcmRpbmF0ZWAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgY29uc3QgYm9keVZlY3RvciA9IFBvc2VTZXQuZ2V0Qm9keVZlY3Rvcihwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZSk7XG4gICAgaWYgKCFib2R5VmVjdG9yKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yYCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1dpdGhXb3JsZENvb3JkaW5hdGVcbiAgICAgICk7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgaWYgKFxuICAgICAgcmVzdWx0cy5sZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkICYmXG4gICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkXG4gICAgKSB7XG4gICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgIGBbUG9zZVNldF0gcHVzaFBvc2UgKCR7dmlkZW9UaW1lTWlsaXNlY29uZHN9KSAtIENvdWxkIG5vdCBnZXQgdGhlIGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCkge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBsZWZ0IGhhbmQgbGFuZG1hcmtzYCxcbiAgICAgICAgcmVzdWx0c1xuICAgICAgKTtcbiAgICB9IGVsc2UgaWYgKHJlc3VsdHMucmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQpIHtcbiAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgYFtQb3NlU2V0XSBwdXNoUG9zZSAoJHt2aWRlb1RpbWVNaWxpc2Vjb25kc30pIC0gQ291bGQgbm90IGdldCB0aGUgcmlnaHQgaGFuZCBsYW5kbWFya3NgLFxuICAgICAgICByZXN1bHRzXG4gICAgICApO1xuICAgIH1cblxuICAgIGNvbnN0IGhhbmRWZWN0b3IgPSBQb3NlU2V0LmdldEhhbmRWZWN0b3IoXG4gICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3NcbiAgICApO1xuICAgIGlmICghaGFuZFZlY3Rvcikge1xuICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICBgW1Bvc2VTZXRdIHB1c2hQb3NlICgke3ZpZGVvVGltZU1pbGlzZWNvbmRzfSkgLSBDb3VsZCBub3QgZ2V0IHRoZSBoYW5kIHZlY3RvcmAsXG4gICAgICAgIHJlc3VsdHNcbiAgICAgICk7XG4gICAgfVxuXG4gICAgY29uc3QgcG9zZTogUG9zZVNldEl0ZW0gPSB7XG4gICAgICB0aW1lTWlsaXNlY29uZHM6IHZpZGVvVGltZU1pbGlzZWNvbmRzLFxuICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogLTEsXG4gICAgICBwb3NlOiBwb3NlTGFuZG1hcmtzV2l0aFdvcmxkQ29vcmRpbmF0ZS5tYXAoKHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsueCxcbiAgICAgICAgICB3b3JsZENvb3JkaW5hdGVMYW5kbWFyay55LFxuICAgICAgICAgIHdvcmxkQ29vcmRpbmF0ZUxhbmRtYXJrLnosXG4gICAgICAgICAgd29ybGRDb29yZGluYXRlTGFuZG1hcmsudmlzaWJpbGl0eSxcbiAgICAgICAgXTtcbiAgICAgIH0pLFxuICAgICAgbGVmdEhhbmQ6IHJlc3VsdHMubGVmdEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIHJpZ2h0SGFuZDogcmVzdWx0cy5yaWdodEhhbmRMYW5kbWFya3M/Lm1hcCgobm9ybWFsaXplZExhbmRtYXJrKSA9PiB7XG4gICAgICAgIHJldHVybiBbXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLngsXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnksXG4gICAgICAgICAgbm9ybWFsaXplZExhbmRtYXJrLnosXG4gICAgICAgIF07XG4gICAgICB9KSxcbiAgICAgIGJvZHlWZWN0b3I6IGJvZHlWZWN0b3IsXG4gICAgICBoYW5kVmVjdG9yOiBoYW5kVmVjdG9yLFxuICAgICAgZnJhbWVJbWFnZURhdGFVcmw6IGZyYW1lSW1hZ2VEYXRhVXJsLFxuICAgICAgcG9zZUltYWdlRGF0YVVybDogcG9zZUltYWdlRGF0YVVybCxcbiAgICAgIGZhY2VGcmFtZUltYWdlRGF0YVVybDogZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLFxuICAgICAgZXh0ZW5kZWREYXRhOiB7fSxcbiAgICB9O1xuXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIC8vIOWJjeWbnuOBruODneODvOOCuuOBqOOBrumhnuS8vOaAp+OCkuODgeOCp+ODg+OCr1xuICAgICAgY29uc3QgbGFzdFBvc2UgPSB0aGlzLnBvc2VzW3RoaXMucG9zZXMubGVuZ3RoIC0gMV07XG5cbiAgICAgIGNvbnN0IGlzU2ltaWxhckJvZHlQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJCb2R5UG9zZShcbiAgICAgICAgbGFzdFBvc2UuYm9keVZlY3RvcixcbiAgICAgICAgcG9zZS5ib2R5VmVjdG9yXG4gICAgICApO1xuXG4gICAgICBsZXQgaXNTaW1pbGFySGFuZFBvc2UgPSB0cnVlO1xuICAgICAgaWYgKGxhc3RQb3NlLmhhbmRWZWN0b3IgJiYgcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGlzU2ltaWxhckhhbmRQb3NlID0gUG9zZVNldC5pc1NpbWlsYXJIYW5kUG9zZShcbiAgICAgICAgICBsYXN0UG9zZS5oYW5kVmVjdG9yLFxuICAgICAgICAgIHBvc2UuaGFuZFZlY3RvclxuICAgICAgICApO1xuICAgICAgfSBlbHNlIGlmICghbGFzdFBvc2UuaGFuZFZlY3RvciAmJiBwb3NlLmhhbmRWZWN0b3IpIHtcbiAgICAgICAgaXNTaW1pbGFySGFuZFBvc2UgPSBmYWxzZTtcbiAgICAgIH1cblxuICAgICAgaWYgKGlzU2ltaWxhckJvZHlQb3NlICYmIGlzU2ltaWxhckhhbmRQb3NlKSB7XG4gICAgICAgIC8vIOi6q+S9k+ODu+aJi+OBqOOCguOBq+mhnuS8vOODneODvOOCuuOBquOCieOBsOOCueOCreODg+ODl1xuICAgICAgICByZXR1cm47XG4gICAgICB9XG5cbiAgICAgIC8vIOWJjeWbnuOBruODneODvOOCuuOBruaMgee2muaZgumWk+OCkuioreWumlxuICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICB2aWRlb1RpbWVNaWxpc2Vjb25kcyAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgIHRoaXMucG9zZXNbdGhpcy5wb3Nlcy5sZW5ndGggLSAxXS5kdXJhdGlvbk1pbGlzZWNvbmRzID1cbiAgICAgICAgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHM7XG4gICAgfVxuXG4gICAgdGhpcy5wb3Nlcy5wdXNoKHBvc2UpO1xuXG4gICAgcmV0dXJuIHBvc2U7XG4gIH1cblxuICBhc3luYyBmaW5hbGl6ZSgpIHtcbiAgICBpZiAoMCA9PSB0aGlzLnBvc2VzLmxlbmd0aCkge1xuICAgICAgdGhpcy5pc0ZpbmFsaXplZCA9IHRydWU7XG4gICAgICByZXR1cm47XG4gICAgfVxuXG4gICAgLy8g5pyA5b6M44Gu44Od44O844K644Gu5oyB57aa5pmC6ZaT44KS6Kit5a6aXG4gICAgaWYgKDEgPD0gdGhpcy5wb3Nlcy5sZW5ndGgpIHtcbiAgICAgIGNvbnN0IGxhc3RQb3NlID0gdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdO1xuICAgICAgaWYgKGxhc3RQb3NlLmR1cmF0aW9uTWlsaXNlY29uZHMgPT0gLTEpIHtcbiAgICAgICAgY29uc3QgcG9zZUR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHRoaXMudmlkZW9NZXRhZGF0YS5kdXJhdGlvbiAtIGxhc3RQb3NlLnRpbWVNaWxpc2Vjb25kcztcbiAgICAgICAgdGhpcy5wb3Nlc1t0aGlzLnBvc2VzLmxlbmd0aCAtIDFdLmR1cmF0aW9uTWlsaXNlY29uZHMgPVxuICAgICAgICAgIHBvc2VEdXJhdGlvbk1pbGlzZWNvbmRzO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOmHjeikh+ODneODvOOCuuOCkumZpOWOu1xuICAgIHRoaXMucmVtb3ZlRHVwbGljYXRlZFBvc2VzKCk7XG5cbiAgICAvLyDmnIDliJ3jga7jg53jg7zjgrrjgpLpmaTljrtcbiAgICB0aGlzLnBvc2VzLnNoaWZ0KCk7XG5cbiAgICAvLyDnlLvlg4/jga7jg57jg7zjgrjjg7PjgpLlj5blvpdcbiAgICBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGZpbmFsaXplIC0gRGV0ZWN0aW5nIGltYWdlIG1hcmdpbnMuLi5gKTtcbiAgICBsZXQgaW1hZ2VUcmltbWluZzpcbiAgICAgIHwge1xuICAgICAgICAgIG1hcmdpblRvcDogbnVtYmVyO1xuICAgICAgICAgIG1hcmdpbkJvdHRvbTogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE5ldzogbnVtYmVyO1xuICAgICAgICAgIGhlaWdodE9sZDogbnVtYmVyO1xuICAgICAgICAgIHdpZHRoOiBudW1iZXI7XG4gICAgICAgIH1cbiAgICAgIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLmxvYWRCeURhdGFVcmwocG9zZS5mcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGNvbnN0IG1hcmdpbkNvbG9yID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldE1hcmdpbkNvbG9yKCk7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVjdGVkIG1hcmdpbiBjb2xvci4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzLFxuICAgICAgICBtYXJnaW5Db2xvclxuICAgICAgKTtcbiAgICAgIGlmIChtYXJnaW5Db2xvciA9PT0gbnVsbCkgY29udGludWU7XG4gICAgICBpZiAobWFyZ2luQ29sb3IgIT09IHRoaXMuSU1BR0VfTUFSR0lOX1RSSU1NSU5HX0NPTE9SKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuICAgICAgY29uc3QgdHJpbW1lZCA9IGF3YWl0IGltYWdlVHJpbW1lci50cmltTWFyZ2luKFxuICAgICAgICBtYXJnaW5Db2xvcixcbiAgICAgICAgdGhpcy5JTUFHRV9NQVJHSU5fVFJJTU1JTkdfRElGRl9USFJFU0hPTERcbiAgICAgICk7XG4gICAgICBpZiAoIXRyaW1tZWQpIGNvbnRpbnVlO1xuICAgICAgaW1hZ2VUcmltbWluZyA9IHRyaW1tZWQ7XG4gICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgYFtQb3NlU2V0XSBmaW5hbGl6ZSAtIERldGVybWluZWQgaW1hZ2UgdHJpbW1pbmcgcG9zaXRpb25zLi4uYCxcbiAgICAgICAgdHJpbW1lZFxuICAgICAgKTtcbiAgICAgIGJyZWFrO1xuICAgIH1cblxuICAgIC8vIOeUu+WDj+OCkuaVtOW9olxuICAgIGZvciAoY29uc3QgcG9zZSBvZiB0aGlzLnBvc2VzKSB7XG4gICAgICBsZXQgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgaWYgKCFwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsIHx8ICFwb3NlLnBvc2VJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgY29udGludWU7XG4gICAgICB9XG5cbiAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gUHJvY2Vzc2luZyBpbWFnZS4uLmAsXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpO1xuXG4gICAgICBpZiAoaW1hZ2VUcmltbWluZykge1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIuY3JvcChcbiAgICAgICAgICAwLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcubWFyZ2luVG9wLFxuICAgICAgICAgIGltYWdlVHJpbW1pbmcud2lkdGgsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5oZWlnaHROZXdcbiAgICAgICAgKTtcbiAgICAgIH1cblxuICAgICAgYXdhaXQgaW1hZ2VUcmltbWVyLnJlcGxhY2VDb2xvcihcbiAgICAgICAgdGhpcy5JTUFHRV9CQUNLR1JPVU5EX1JFUExBQ0VfU1JDX0NPTE9SLFxuICAgICAgICB0aGlzLklNQUdFX0JBQ0tHUk9VTkRfUkVQTEFDRV9EU1RfQ09MT1IsXG4gICAgICAgIHRoaXMuSU1BR0VfQkFDS0dST1VORF9SRVBMQUNFX0RJRkZfVEhSRVNIT0xEXG4gICAgICApO1xuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIGxldCBuZXdEYXRhVXJsID0gYXdhaXQgaW1hZ2VUcmltbWVyLmdldERhdGFVcmwoXG4gICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2UvanBlZycgfHwgdGhpcy5JTUFHRV9NSU1FID09PSAnaW1hZ2Uvd2VicCdcbiAgICAgICAgICA/IHRoaXMuSU1BR0VfUVVBTElUWVxuICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICApO1xuICAgICAgaWYgKCFuZXdEYXRhVXJsKSB7XG4gICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICBgW1Bvc2VTZXRdIGZpbmFsaXplIC0gQ291bGQgbm90IGdldCB0aGUgbmV3IGRhdGF1cmwgZm9yIGZyYW1lIGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UuZnJhbWVJbWFnZURhdGFVcmwgPSBuZXdEYXRhVXJsO1xuXG4gICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDjg53jg7zjgrrjg5fjg6zjg5Pjg6Xjg7znlLvlg49cbiAgICAgIGltYWdlVHJpbW1lciA9IG5ldyBJbWFnZVRyaW1tZXIoKTtcbiAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5sb2FkQnlEYXRhVXJsKHBvc2UucG9zZUltYWdlRGF0YVVybCk7XG5cbiAgICAgIGlmIChpbWFnZVRyaW1taW5nKSB7XG4gICAgICAgIGF3YWl0IGltYWdlVHJpbW1lci5jcm9wKFxuICAgICAgICAgIDAsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy5tYXJnaW5Ub3AsXG4gICAgICAgICAgaW1hZ2VUcmltbWluZy53aWR0aCxcbiAgICAgICAgICBpbWFnZVRyaW1taW5nLmhlaWdodE5ld1xuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICBhd2FpdCBpbWFnZVRyaW1tZXIucmVzaXplV2l0aEZpdCh7XG4gICAgICAgIHdpZHRoOiB0aGlzLklNQUdFX1dJRFRILFxuICAgICAgfSk7XG5cbiAgICAgIG5ld0RhdGFVcmwgPSBhd2FpdCBpbWFnZVRyaW1tZXIuZ2V0RGF0YVVybChcbiAgICAgICAgdGhpcy5JTUFHRV9NSU1FLFxuICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgID8gdGhpcy5JTUFHRV9RVUFMSVRZXG4gICAgICAgICAgOiB1bmRlZmluZWRcbiAgICAgICk7XG4gICAgICBpZiAoIW5ld0RhdGFVcmwpIHtcbiAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgcG9zZSBwcmV2aWV3IGltYWdlYFxuICAgICAgICApO1xuICAgICAgICBjb250aW51ZTtcbiAgICAgIH1cbiAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybCA9IG5ld0RhdGFVcmw7XG5cbiAgICAgIGlmIChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCkge1xuICAgICAgICAvLyDnlLvlg4/jgpLmlbTlvaIgLSDpoZTjg5Xjg6zjg7zjg6DnlLvlg49cbiAgICAgICAgaW1hZ2VUcmltbWVyID0gbmV3IEltYWdlVHJpbW1lcigpO1xuICAgICAgICBhd2FpdCBpbWFnZVRyaW1tZXIubG9hZEJ5RGF0YVVybChwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybCk7XG5cbiAgICAgICAgbmV3RGF0YVVybCA9IGF3YWl0IGltYWdlVHJpbW1lci5nZXREYXRhVXJsKFxuICAgICAgICAgIHRoaXMuSU1BR0VfTUlNRSxcbiAgICAgICAgICB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS9qcGVnJyB8fCB0aGlzLklNQUdFX01JTUUgPT09ICdpbWFnZS93ZWJwJ1xuICAgICAgICAgICAgPyB0aGlzLklNQUdFX1FVQUxJVFlcbiAgICAgICAgICAgIDogdW5kZWZpbmVkXG4gICAgICAgICk7XG4gICAgICAgIGlmICghbmV3RGF0YVVybCkge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZVNldF0gZmluYWxpemUgLSBDb3VsZCBub3QgZ2V0IHRoZSBuZXcgZGF0YXVybCBmb3IgZmFjZSBmcmFtZSBpbWFnZWBcbiAgICAgICAgICApO1xuICAgICAgICAgIGNvbnRpbnVlO1xuICAgICAgICB9XG4gICAgICAgIHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsID0gbmV3RGF0YVVybDtcbiAgICAgIH1cbiAgICB9XG5cbiAgICB0aGlzLmlzRmluYWxpemVkID0gdHJ1ZTtcbiAgfVxuXG4gIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcygpOiB2b2lkIHtcbiAgICAvLyDlhajjg53jg7zjgrrjgpLmr5TovIPjgZfjgabpoZ7kvLzjg53jg7zjgrrjgpLliYrpmaRcbiAgICBjb25zdCBuZXdQb3NlczogUG9zZVNldEl0ZW1bXSA9IFtdO1xuICAgIGZvciAoY29uc3QgcG9zZUEgb2YgdGhpcy5wb3Nlcykge1xuICAgICAgbGV0IGlzRHVwbGljYXRlZCA9IGZhbHNlO1xuICAgICAgZm9yIChjb25zdCBwb3NlQiBvZiBuZXdQb3Nlcykge1xuICAgICAgICBjb25zdCBpc1NpbWlsYXJCb2R5UG9zZSA9IFBvc2VTZXQuaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgICAgICAgcG9zZUEuYm9keVZlY3RvcixcbiAgICAgICAgICBwb3NlQi5ib2R5VmVjdG9yXG4gICAgICAgICk7XG4gICAgICAgIGNvbnN0IGlzU2ltaWxhckhhbmRQb3NlID1cbiAgICAgICAgICBwb3NlQS5oYW5kVmVjdG9yICYmIHBvc2VCLmhhbmRWZWN0b3JcbiAgICAgICAgICAgID8gUG9zZVNldC5pc1NpbWlsYXJIYW5kUG9zZShwb3NlQS5oYW5kVmVjdG9yLCBwb3NlQi5oYW5kVmVjdG9yKVxuICAgICAgICAgICAgOiBmYWxzZTtcblxuICAgICAgICBpZiAoaXNTaW1pbGFyQm9keVBvc2UgJiYgaXNTaW1pbGFySGFuZFBvc2UpIHtcbiAgICAgICAgICAvLyDouqvkvZPjg7vmiYvjgajjgoLjgavpoZ7kvLzjg53jg7zjgrrjgarjgonjgbBcbiAgICAgICAgICBpc0R1cGxpY2F0ZWQgPSB0cnVlO1xuICAgICAgICAgIGJyZWFrO1xuICAgICAgICB9XG4gICAgICB9XG5cbiAgICAgIGlmIChpc0R1cGxpY2F0ZWQpIGNvbnRpbnVlO1xuXG4gICAgICBuZXdQb3Nlcy5wdXNoKHBvc2VBKTtcbiAgICB9XG5cbiAgICBjb25zb2xlLmluZm8oXG4gICAgICBgW1Bvc2VTZXRdIHJlbW92ZUR1cGxpY2F0ZWRQb3NlcyAtIFJlZHVjZWQgJHt0aGlzLnBvc2VzLmxlbmd0aH0gcG9zZXMgLT4gJHtuZXdQb3Nlcy5sZW5ndGh9IHBvc2VzYFxuICAgICk7XG4gICAgdGhpcy5wb3NlcyA9IG5ld1Bvc2VzO1xuICB9XG5cbiAgZ2V0U2ltaWxhclBvc2VzKFxuICAgIHJlc3VsdHM6IFJlc3VsdHMsXG4gICAgdGhyZXNob2xkOiBudW1iZXIgPSAwLjksXG4gICAgdGFyZ2V0UmFuZ2U6ICdhbGwnIHwgJ2JvZHlQb3NlJyB8ICdoYW5kUG9zZScgPSAnYWxsJ1xuICApOiBTaW1pbGFyUG9zZUl0ZW1bXSB7XG4gICAgLy8g6Lqr5L2T44Gu44OZ44Kv44OI44Or44KS5Y+W5b6XXG4gICAgbGV0IGJvZHlWZWN0b3I6IEJvZHlWZWN0b3I7XG4gICAgdHJ5IHtcbiAgICAgIGJvZHlWZWN0b3IgPSBQb3NlU2V0LmdldEJvZHlWZWN0b3IoKHJlc3VsdHMgYXMgYW55KS5lYSk7XG4gICAgfSBjYXRjaCAoZSkge1xuICAgICAgY29uc29sZS5lcnJvcihgW1Bvc2VTZXRdIGdldFNpbWlsYXJQb3NlcyAtIEVycm9yIG9jY3VycmVkYCwgZSwgcmVzdWx0cyk7XG4gICAgICByZXR1cm4gW107XG4gICAgfVxuICAgIGlmICghYm9keVZlY3Rvcikge1xuICAgICAgdGhyb3cgJ0NvdWxkIG5vdCBnZXQgdGhlIGJvZHkgdmVjdG9yJztcbiAgICB9XG5cbiAgICAvLyDmiYvmjIfjga7jg5njgq/jg4jjg6vjgpLlj5blvpdcbiAgICBsZXQgaGFuZFZlY3RvcjogSGFuZFZlY3RvcjtcbiAgICBpZiAodGFyZ2V0UmFuZ2UgPT09ICdhbGwnIHx8IHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnKSB7XG4gICAgICBoYW5kVmVjdG9yID0gUG9zZVNldC5nZXRIYW5kVmVjdG9yKFxuICAgICAgICByZXN1bHRzLmxlZnRIYW5kTGFuZG1hcmtzLFxuICAgICAgICByZXN1bHRzLnJpZ2h0SGFuZExhbmRtYXJrc1xuICAgICAgKTtcbiAgICAgIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJyAmJiAhaGFuZFZlY3Rvcikge1xuICAgICAgICB0aHJvdyAnQ291bGQgbm90IGdldCB0aGUgaGFuZCB2ZWN0b3InO1xuICAgICAgfVxuICAgIH1cblxuICAgIC8vIOWQhOODneODvOOCuuOBqOODmeOCr+ODiOODq+OCkuavlOi8g1xuICAgIGNvbnN0IHBvc2VzID0gW107XG4gICAgZm9yIChjb25zdCBwb3NlIG9mIHRoaXMucG9zZXMpIHtcbiAgICAgIGlmIChcbiAgICAgICAgKHRhcmdldFJhbmdlID09PSAnYWxsJyB8fCB0YXJnZXRSYW5nZSA9PT0gJ2JvZHlQb3NlJykgJiZcbiAgICAgICAgIXBvc2UuYm9keVZlY3RvclxuICAgICAgKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2hhbmRQb3NlJyAmJiAhcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGNvbnRpbnVlO1xuICAgICAgfVxuXG4gICAgICAvKmNvbnNvbGUubG9nKFxuICAgICAgICAnW1Bvc2VTZXRdIGdldFNpbWlsYXJQb3NlcyAtICcsXG4gICAgICAgIHRoaXMuZ2V0VmlkZW9OYW1lKCksXG4gICAgICAgIHBvc2UudGltZU1pbGlzZWNvbmRzXG4gICAgICApOyovXG5cbiAgICAgIC8vIOi6q+S9k+OBruODneODvOOCuuOBrumhnuS8vOW6puOCkuWPluW+l1xuICAgICAgbGV0IGJvZHlTaW1pbGFyaXR5OiBudW1iZXI7XG4gICAgICBpZiAoYm9keVZlY3RvciAmJiBwb3NlLmJvZHlWZWN0b3IpIHtcbiAgICAgICAgYm9keVNpbWlsYXJpdHkgPSBQb3NlU2V0LmdldEJvZHlQb3NlU2ltaWxhcml0eShcbiAgICAgICAgICBwb3NlLmJvZHlWZWN0b3IsXG4gICAgICAgICAgYm9keVZlY3RvclxuICAgICAgICApO1xuICAgICAgfVxuXG4gICAgICAvLyDmiYvmjIfjga7jg53jg7zjgrrjga7poZ7kvLzluqbjgpLlj5blvpdcbiAgICAgIGxldCBoYW5kU2ltaWxhcml0eTogbnVtYmVyO1xuICAgICAgaWYgKGhhbmRWZWN0b3IgJiYgcG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgIGhhbmRTaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRIYW5kU2ltaWxhcml0eShwb3NlLmhhbmRWZWN0b3IsIGhhbmRWZWN0b3IpO1xuICAgICAgfVxuXG4gICAgICAvLyDliKTlrppcbiAgICAgIGxldCBzaW1pbGFyaXR5OiBudW1iZXIsXG4gICAgICAgIGlzU2ltaWxhciA9IGZhbHNlO1xuICAgICAgaWYgKHRhcmdldFJhbmdlID09PSAnYWxsJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gTWF0aC5tYXgoYm9keVNpbWlsYXJpdHkgPz8gMCwgaGFuZFNpbWlsYXJpdHkgPz8gMCk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gYm9keVNpbWlsYXJpdHkgfHwgdGhyZXNob2xkIDw9IGhhbmRTaW1pbGFyaXR5KSB7XG4gICAgICAgICAgaXNTaW1pbGFyID0gdHJ1ZTtcbiAgICAgICAgfVxuICAgICAgfSBlbHNlIGlmICh0YXJnZXRSYW5nZSA9PT0gJ2JvZHlQb3NlJykge1xuICAgICAgICBzaW1pbGFyaXR5ID0gYm9keVNpbWlsYXJpdHk7XG4gICAgICAgIGlmICh0aHJlc2hvbGQgPD0gYm9keVNpbWlsYXJpdHkpIHtcbiAgICAgICAgICBpc1NpbWlsYXIgPSB0cnVlO1xuICAgICAgICB9XG4gICAgICB9IGVsc2UgaWYgKHRhcmdldFJhbmdlID09PSAnaGFuZFBvc2UnKSB7XG4gICAgICAgIHNpbWlsYXJpdHkgPSBoYW5kU2ltaWxhcml0eTtcbiAgICAgICAgaWYgKHRocmVzaG9sZCA8PSBoYW5kU2ltaWxhcml0eSkge1xuICAgICAgICAgIGlzU2ltaWxhciA9IHRydWU7XG4gICAgICAgIH1cbiAgICAgIH1cblxuICAgICAgaWYgKCFpc1NpbWlsYXIpIGNvbnRpbnVlO1xuXG4gICAgICAvLyDntZDmnpzjgbjov73liqBcbiAgICAgIHBvc2VzLnB1c2goe1xuICAgICAgICAuLi5wb3NlLFxuICAgICAgICBzaW1pbGFyaXR5OiBzaW1pbGFyaXR5LFxuICAgICAgICBib2R5UG9zZVNpbWlsYXJpdHk6IGJvZHlTaW1pbGFyaXR5LFxuICAgICAgICBoYW5kUG9zZVNpbWlsYXJpdHk6IGhhbmRTaW1pbGFyaXR5LFxuICAgICAgfSBhcyBTaW1pbGFyUG9zZUl0ZW0pO1xuICAgIH1cblxuICAgIHJldHVybiBwb3NlcztcbiAgfVxuXG4gIHN0YXRpYyBnZXRCb2R5VmVjdG9yKFxuICAgIHBvc2VMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W11cbiAgKTogQm9keVZlY3RvciB8IHVuZGVmaW5lZCB7XG4gICAgcmV0dXJuIHtcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueCAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueSxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9XUklTVF0ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5SSUdIVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBbXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnkgLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfRUxCT1ddLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuUklHSFRfU0hPVUxERVJdLnosXG4gICAgICBdLFxuICAgICAgbGVmdFdyaXN0VG9MZWZ0RWxib3c6IFtcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1dSSVNUXS54IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLngsXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9XUklTVF0ueSAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55LFxuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfV1JJU1RdLnogLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueixcbiAgICAgIF0sXG4gICAgICBsZWZ0RWxib3dUb0xlZnRTaG91bGRlcjogW1xuICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfRUxCT1ddLnggLVxuICAgICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9TSE9VTERFUl0ueCxcbiAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX0VMQk9XXS55IC1cbiAgICAgICAgICBwb3NlTGFuZG1hcmtzW1BPU0VfTEFORE1BUktTLkxFRlRfU0hPVUxERVJdLnksXG4gICAgICAgIHBvc2VMYW5kbWFya3NbUE9TRV9MQU5ETUFSS1MuTEVGVF9FTEJPV10ueiAtXG4gICAgICAgICAgcG9zZUxhbmRtYXJrc1tQT1NFX0xBTkRNQVJLUy5MRUZUX1NIT1VMREVSXS56LFxuICAgICAgXSxcbiAgICB9O1xuICB9XG5cbiAgc3RhdGljIGdldEhhbmRWZWN0b3IoXG4gICAgbGVmdEhhbmRMYW5kbWFya3M6IHsgeDogbnVtYmVyOyB5OiBudW1iZXI7IHo6IG51bWJlciB9W10sXG4gICAgcmlnaHRIYW5kTGFuZG1hcmtzOiB7IHg6IG51bWJlcjsgeTogbnVtYmVyOyB6OiBudW1iZXIgfVtdXG4gICk6IEhhbmRWZWN0b3IgfCB1bmRlZmluZWQge1xuICAgIGlmIChcbiAgICAgIChyaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwKSAmJlxuICAgICAgKGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwKVxuICAgICkge1xuICAgICAgcmV0dXJuIHVuZGVmaW5lZDtcbiAgICB9XG5cbiAgICByZXR1cm4ge1xuICAgICAgLy8g5Y+z5omLIC0g6Kaq5oyHXG4gICAgICByaWdodFRodW1iVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s0XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzNdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s0XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s0XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzNdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1szXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzJdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1szXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzJdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1szXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzJdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgICByaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s4XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzddLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s4XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzddLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s4XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzddLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzZdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzZdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1s3XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzZdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g5Lit5oyHXG4gICAgICByaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTJdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTFdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMl0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzEyXS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzExXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IHJpZ2h0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzExXS54IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzEwXS54LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTFdLnkgLSByaWdodEhhbmRMYW5kbWFya3NbMTBdLnksXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxMV0ueiAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxMF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAgIHJpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDpcbiAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgcmlnaHRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTZdLnggLSByaWdodEhhbmRMYW5kbWFya3NbMTVdLngsXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNl0ueSAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueSxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE2XS56IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIHJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxNV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxNF0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE1XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE0XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTVdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTRdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5Y+z5omLIC0g5bCP5oyHXG4gICAgICByaWdodFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1syMF0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzIwXS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMjBdLnogLSByaWdodEhhbmRMYW5kbWFya3NbMTldLnosXG4gICAgICAgICAgICBdLFxuICAgICAgcmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OlxuICAgICAgICByaWdodEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCByaWdodEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIHJpZ2h0SGFuZExhbmRtYXJrc1sxOV0ueCAtIHJpZ2h0SGFuZExhbmRtYXJrc1sxOF0ueCxcbiAgICAgICAgICAgICAgcmlnaHRIYW5kTGFuZG1hcmtzWzE5XS55IC0gcmlnaHRIYW5kTGFuZG1hcmtzWzE4XS55LFxuICAgICAgICAgICAgICByaWdodEhhbmRMYW5kbWFya3NbMTldLnogLSByaWdodEhhbmRMYW5kbWFya3NbMThdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g6Kaq5oyHXG4gICAgICBsZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzRdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1szXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s0XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbM10ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbNF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzNdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1syXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1szXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMl0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbM10ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzJdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzhdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1s3XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s4XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbN10ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbOF0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzddLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzddLnggLSBsZWZ0SGFuZExhbmRtYXJrc1s2XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1s3XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbNl0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbN10ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzZdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5Lit5oyHXG4gICAgICBsZWZ0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMl0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzExXS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMl0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzExXS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxMl0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzExXS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDpcbiAgICAgICAgbGVmdEhhbmRMYW5kbWFya3MgPT09IHVuZGVmaW5lZCB8fCBsZWZ0SGFuZExhbmRtYXJrcy5sZW5ndGggPT09IDBcbiAgICAgICAgICA/IG51bGxcbiAgICAgICAgICA6IFtcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTFdLnggLSBsZWZ0SGFuZExhbmRtYXJrc1sxMF0ueCxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTFdLnkgLSBsZWZ0SGFuZExhbmRtYXJrc1sxMF0ueSxcbiAgICAgICAgICAgICAgbGVmdEhhbmRMYW5kbWFya3NbMTFdLnogLSBsZWZ0SGFuZExhbmRtYXJrc1sxMF0ueixcbiAgICAgICAgICAgIF0sXG4gICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OlxuICAgICAgICBsZWZ0SGFuZExhbmRtYXJrcyA9PT0gdW5kZWZpbmVkIHx8IGxlZnRIYW5kTGFuZG1hcmtzLmxlbmd0aCA9PT0gMFxuICAgICAgICAgID8gbnVsbFxuICAgICAgICAgIDogW1xuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueCAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS54LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueSAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS55LFxuICAgICAgICAgICAgICBsZWZ0SGFuZExhbmRtYXJrc1sxNl0ueiAtIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS56LFxuICAgICAgICAgICAgXSxcbiAgICAgIGxlZnRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTRdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTRdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE1XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTRdLnosXG4gICAgICAgICAgICBdLFxuICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzIwXS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMTldLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzIwXS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMTldLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzIwXS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMTldLnosXG4gICAgICAgICAgICBdLFxuICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6XG4gICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzID09PSB1bmRlZmluZWQgfHwgbGVmdEhhbmRMYW5kbWFya3MubGVuZ3RoID09PSAwXG4gICAgICAgICAgPyBudWxsXG4gICAgICAgICAgOiBbXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS54IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLngsXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS55IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLnksXG4gICAgICAgICAgICAgIGxlZnRIYW5kTGFuZG1hcmtzWzE5XS56IC0gbGVmdEhhbmRMYW5kbWFya3NbMThdLnosXG4gICAgICAgICAgICBdLFxuICAgIH07XG4gIH1cblxuICBzdGF0aWMgaXNTaW1pbGFyQm9keVBvc2UoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3IsXG4gICAgdGhyZXNob2xkID0gMC44XG4gICk6IGJvb2xlYW4ge1xuICAgIGxldCBpc1NpbWlsYXIgPSBmYWxzZTtcbiAgICBjb25zdCBzaW1pbGFyaXR5ID0gUG9zZVNldC5nZXRCb2R5UG9zZVNpbWlsYXJpdHkoYm9keVZlY3RvckEsIGJvZHlWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQpIGlzU2ltaWxhciA9IHRydWU7XG5cbiAgICAvLyBjb25zb2xlLmxvZyhgW1Bvc2VTZXRdIGlzU2ltaWxhclBvc2VgLCBpc1NpbWlsYXIsIHNpbWlsYXJpdHkpO1xuXG4gICAgcmV0dXJuIGlzU2ltaWxhcjtcbiAgfVxuXG4gIHN0YXRpYyBnZXRCb2R5UG9zZVNpbWlsYXJpdHkoXG4gICAgYm9keVZlY3RvckE6IEJvZHlWZWN0b3IsXG4gICAgYm9keVZlY3RvckI6IEJvZHlWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXMgPSB7XG4gICAgICBsZWZ0V3Jpc3RUb0xlZnRFbGJvdzogY29zU2ltaWxhcml0eShcbiAgICAgICAgYm9keVZlY3RvckEubGVmdFdyaXN0VG9MZWZ0RWxib3csXG4gICAgICAgIGJvZHlWZWN0b3JCLmxlZnRXcmlzdFRvTGVmdEVsYm93XG4gICAgICApLFxuICAgICAgbGVmdEVsYm93VG9MZWZ0U2hvdWxkZXI6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLmxlZnRFbGJvd1RvTGVmdFNob3VsZGVyLFxuICAgICAgICBib2R5VmVjdG9yQi5sZWZ0RWxib3dUb0xlZnRTaG91bGRlclxuICAgICAgKSxcbiAgICAgIHJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3c6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgIGJvZHlWZWN0b3JBLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3csXG4gICAgICAgIGJvZHlWZWN0b3JCLnJpZ2h0V3Jpc3RUb1JpZ2h0RWxib3dcbiAgICAgICksXG4gICAgICByaWdodEVsYm93VG9SaWdodFNob3VsZGVyOiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICBib2R5VmVjdG9yQS5yaWdodEVsYm93VG9SaWdodFNob3VsZGVyLFxuICAgICAgICBib2R5VmVjdG9yQi5yaWdodEVsYm93VG9SaWdodFNob3VsZGVyXG4gICAgICApLFxuICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNTdW0gPSBPYmplY3QudmFsdWVzKGNvc1NpbWlsYXJpdGllcykucmVkdWNlKFxuICAgICAgKHN1bSwgdmFsdWUpID0+IHN1bSArIHZhbHVlLFxuICAgICAgMFxuICAgICk7XG4gICAgcmV0dXJuIGNvc1NpbWlsYXJpdGllc1N1bSAvIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllcykubGVuZ3RoO1xuICB9XG5cbiAgc3RhdGljIGlzU2ltaWxhckhhbmRQb3NlKFxuICAgIGhhbmRWZWN0b3JBOiBIYW5kVmVjdG9yLFxuICAgIGhhbmRWZWN0b3JCOiBIYW5kVmVjdG9yLFxuICAgIHRocmVzaG9sZCA9IDAuNzVcbiAgKTogYm9vbGVhbiB7XG4gICAgY29uc3Qgc2ltaWxhcml0eSA9IFBvc2VTZXQuZ2V0SGFuZFNpbWlsYXJpdHkoaGFuZFZlY3RvckEsIGhhbmRWZWN0b3JCKTtcbiAgICBpZiAoc2ltaWxhcml0eSA9PT0gLTEpIHtcbiAgICAgIHJldHVybiB0cnVlO1xuICAgIH1cbiAgICByZXR1cm4gc2ltaWxhcml0eSA+PSB0aHJlc2hvbGQ7XG4gIH1cblxuICBzdGF0aWMgZ2V0SGFuZFNpbWlsYXJpdHkoXG4gICAgaGFuZFZlY3RvckE6IEhhbmRWZWN0b3IsXG4gICAgaGFuZFZlY3RvckI6IEhhbmRWZWN0b3JcbiAgKTogbnVtYmVyIHtcbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgPVxuICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsIHx8XG4gICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIHJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5Y+z5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICByaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgcmlnaHRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOWPs+aJiyAtIOS4reaMh1xuICAgICAgICAgICAgcmlnaHRNaWRkbGVGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0TWlkZGxlRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEucmlnaHRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIHJpZ2h0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5yaWdodFJpbmdGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRSaW5nRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlj7PmiYsgLSDlsI/mjIdcbiAgICAgICAgICAgIHJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLnJpZ2h0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICByaWdodFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLnJpZ2h0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIucmlnaHRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgIH07XG5cbiAgICBjb25zdCBjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCA9XG4gICAgICBoYW5kVmVjdG9yQS5sZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludCA9PT0gbnVsbCB8fFxuICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICAgPyB1bmRlZmluZWRcbiAgICAgICAgOiB7XG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDopqrmjIdcbiAgICAgICAgICAgIGxlZnRUaHVtYlRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iVGlwVG9GaXJzdEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0VGh1bWJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0VGh1bWJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQsXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JCLmxlZnRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5Lq65beu44GX5oyHXG4gICAgICAgICAgICBsZWZ0SW5kZXhGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdEluZGV4RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdEluZGV4RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRJbmRleEZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0SW5kZXhGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIC8vIOW3puaJiyAtIOS4reaMh1xuICAgICAgICAgICAgbGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdE1pZGRsZUZpbmdlclRpcFRvRmlyc3RKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICAgIGxlZnRNaWRkbGVGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludDogY29zU2ltaWxhcml0eShcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckEubGVmdE1pZGRsZUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0TWlkZGxlRmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICAvLyDlt6bmiYsgLSDolqzmjIdcbiAgICAgICAgICAgIGxlZnRSaW5nRmluZ2VyVGlwVG9GaXJzdEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFJpbmdGaW5nZXJUaXBUb0ZpcnN0Sm9pbnRcbiAgICAgICAgICAgICksXG4gICAgICAgICAgICBsZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50OiBjb3NTaW1pbGFyaXR5KFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQS5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UmluZ0ZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgLy8g5bem5omLIC0g5bCP5oyHXG4gICAgICAgICAgICBsZWZ0UGlua3lGaW5nZXJUaXBUb0ZpcnN0Sm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlclRpcFRvRmlyc3RKb2ludCxcbiAgICAgICAgICAgICAgaGFuZFZlY3RvckIubGVmdFBpbmt5RmluZ2VyVGlwVG9GaXJzdEpvaW50XG4gICAgICAgICAgICApLFxuICAgICAgICAgICAgbGVmdFBpbmt5RmluZ2VyRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQ6IGNvc1NpbWlsYXJpdHkoXG4gICAgICAgICAgICAgIGhhbmRWZWN0b3JBLmxlZnRQaW5reUZpbmdlckZpcnN0Sm9pbnRUb1NlY29uZEpvaW50LFxuICAgICAgICAgICAgICBoYW5kVmVjdG9yQi5sZWZ0UGlua3lGaW5nZXJGaXJzdEpvaW50VG9TZWNvbmRKb2ludFxuICAgICAgICAgICAgKSxcbiAgICAgICAgICB9O1xuXG4gICAgLy8g5bem5omL44Gu6aGe5Ly85bqmXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gMDtcbiAgICBpZiAoY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIGNvc1NpbWlsYXJpdGllc1N1bUxlZnRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzTGVmdEhhbmRcbiAgICAgICkucmVkdWNlKChzdW0sIHZhbHVlKSA9PiBzdW0gKyB2YWx1ZSwgMCk7XG4gICAgfVxuXG4gICAgLy8g5Y+z5omL44Gu6aGe5Ly85bqmXG4gICAgbGV0IGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCA9IDA7XG4gICAgaWYgKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCkge1xuICAgICAgY29zU2ltaWxhcml0aWVzU3VtUmlnaHRIYW5kID0gT2JqZWN0LnZhbHVlcyhcbiAgICAgICAgY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kXG4gICAgICApLnJlZHVjZSgoc3VtLCB2YWx1ZSkgPT4gc3VtICsgdmFsdWUsIDApO1xuICAgIH1cblxuICAgIC8vIOWQiOeul+OBleOCjOOBn+mhnuS8vOW6plxuICAgIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQgJiYgY29zU2ltaWxhcml0aWVzTGVmdEhhbmQpIHtcbiAgICAgIHJldHVybiAoXG4gICAgICAgIChjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgKyBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCkgL1xuICAgICAgICAoT2JqZWN0LmtleXMoY29zU2ltaWxhcml0aWVzUmlnaHRIYW5kISkubGVuZ3RoICtcbiAgICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aClcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQpIHtcbiAgICAgIGlmIChcbiAgICAgICAgaGFuZFZlY3RvckIubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgIT09IG51bGwgJiZcbiAgICAgICAgaGFuZFZlY3RvckEubGVmdFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgPT09IG51bGxcbiAgICAgICkge1xuICAgICAgICAvLyBoYW5kVmVjdG9yQiDjgaflt6bmiYvjgYzjgYLjgovjga7jgasgaGFuZFZlY3RvckEg44Gn5bem5omL44GM44Gq44GE5aC05ZCI44CB6aGe5Ly85bqm44KS5rib44KJ44GZXG4gICAgICAgIGNvbnNvbGUubG9nKFxuICAgICAgICAgIGBbUG9zZVNldF0gZ2V0SGFuZFNpbWlsYXJpdHkgLSBBZGp1c3Qgc2ltaWxhcml0eSwgYmVjYXVzZSBsZWZ0IGhhbmQgbm90IGZvdW5kLi4uYFxuICAgICAgICApO1xuICAgICAgICByZXR1cm4gKFxuICAgICAgICAgIGNvc1NpbWlsYXJpdGllc1N1bVJpZ2h0SGFuZCAvXG4gICAgICAgICAgKE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc1JpZ2h0SGFuZCEpLmxlbmd0aCAqIDIpXG4gICAgICAgICk7XG4gICAgICB9XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1SaWdodEhhbmQgL1xuICAgICAgICBPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNSaWdodEhhbmQhKS5sZW5ndGhcbiAgICAgICk7XG4gICAgfSBlbHNlIGlmIChjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCkge1xuICAgICAgaWYgKFxuICAgICAgICBoYW5kVmVjdG9yQi5yaWdodFRodW1iRmlyc3RKb2ludFRvU2Vjb25kSm9pbnQgIT09IG51bGwgJiZcbiAgICAgICAgaGFuZFZlY3RvckEucmlnaHRUaHVtYkZpcnN0Sm9pbnRUb1NlY29uZEpvaW50ID09PSBudWxsXG4gICAgICApIHtcbiAgICAgICAgLy8gaGFuZFZlY3RvckIg44Gn5Y+z5omL44GM44GC44KL44Gu44GrIGhhbmRWZWN0b3JBIOOBp+WPs+aJi+OBjOOBquOBhOWgtOWQiOOAgemhnuS8vOW6puOCkua4m+OCieOBmVxuICAgICAgICBjb25zb2xlLmxvZyhcbiAgICAgICAgICBgW1Bvc2VTZXRdIGdldEhhbmRTaW1pbGFyaXR5IC0gQWRqdXN0IHNpbWlsYXJpdHksIGJlY2F1c2UgcmlnaHQgaGFuZCBub3QgZm91bmQuLi5gXG4gICAgICAgICk7XG4gICAgICAgIHJldHVybiAoXG4gICAgICAgICAgY29zU2ltaWxhcml0aWVzU3VtTGVmdEhhbmQgL1xuICAgICAgICAgIChPYmplY3Qua2V5cyhjb3NTaW1pbGFyaXRpZXNMZWZ0SGFuZCEpLmxlbmd0aCAqIDIpXG4gICAgICAgICk7XG4gICAgICB9XG4gICAgICByZXR1cm4gKFxuICAgICAgICBjb3NTaW1pbGFyaXRpZXNTdW1MZWZ0SGFuZCAvXG4gICAgICAgIE9iamVjdC5rZXlzKGNvc1NpbWlsYXJpdGllc0xlZnRIYW5kISkubGVuZ3RoXG4gICAgICApO1xuICAgIH1cblxuICAgIHJldHVybiAtMTtcbiAgfVxuXG4gIHB1YmxpYyBhc3luYyBnZXRaaXAoKTogUHJvbWlzZTxCbG9iPiB7XG4gICAgY29uc3QganNaaXAgPSBuZXcgSlNaaXAoKTtcbiAgICBqc1ppcC5maWxlKCdwb3Nlcy5qc29uJywgYXdhaXQgdGhpcy5nZXRKc29uKCkpO1xuXG4gICAgY29uc3QgaW1hZ2VGaWxlRXh0ID0gdGhpcy5nZXRGaWxlRXh0ZW5zaW9uQnlNaW1lKHRoaXMuSU1BR0VfTUlNRSk7XG5cbiAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgaWYgKHBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZyYW1lSW1hZ2VEYXRhVXJsLmluZGV4T2YoJ2Jhc2U2NCwnKSArICdiYXNlNjQsJy5sZW5ndGg7XG4gICAgICAgICAgY29uc3QgYmFzZTY0ID0gcG9zZS5mcmFtZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYGZyYW1lLSR7cG9zZS50aW1lTWlsaXNlY29uZHN9LiR7aW1hZ2VGaWxlRXh0fWAsIGJhc2U2NCwge1xuICAgICAgICAgICAgYmFzZTY0OiB0cnVlLFxuICAgICAgICAgIH0pO1xuICAgICAgICB9IGNhdGNoIChlcnJvcikge1xuICAgICAgICAgIGNvbnNvbGUud2FybihcbiAgICAgICAgICAgIGBbUG9zZUV4cG9ydGVyU2VydmljZV0gcHVzaCAtIENvdWxkIG5vdCBwdXNoIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgICAgaWYgKHBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICB0cnkge1xuICAgICAgICAgIGNvbnN0IGluZGV4ID1cbiAgICAgICAgICAgIHBvc2UucG9zZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UucG9zZUltYWdlRGF0YVVybC5zdWJzdHJpbmcoaW5kZXgpO1xuICAgICAgICAgIGpzWmlwLmZpbGUoYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtpbWFnZUZpbGVFeHR9YCwgYmFzZTY0LCB7XG4gICAgICAgICAgICBiYXNlNjQ6IHRydWUsXG4gICAgICAgICAgfSk7XG4gICAgICAgIH0gY2F0Y2ggKGVycm9yKSB7XG4gICAgICAgICAgY29uc29sZS53YXJuKFxuICAgICAgICAgICAgYFtQb3NlRXhwb3J0ZXJTZXJ2aWNlXSBwdXNoIC0gQ291bGQgbm90IHB1c2ggZnJhbWUgaW1hZ2VgLFxuICAgICAgICAgICAgZXJyb3JcbiAgICAgICAgICApO1xuICAgICAgICAgIHRocm93IGVycm9yO1xuICAgICAgICB9XG4gICAgICB9XG4gICAgICBpZiAocG9zZS5mYWNlRnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgdHJ5IHtcbiAgICAgICAgICBjb25zdCBpbmRleCA9XG4gICAgICAgICAgICBwb3NlLmZhY2VGcmFtZUltYWdlRGF0YVVybC5pbmRleE9mKCdiYXNlNjQsJykgKyAnYmFzZTY0LCcubGVuZ3RoO1xuICAgICAgICAgIGNvbnN0IGJhc2U2NCA9IHBvc2UuZmFjZUZyYW1lSW1hZ2VEYXRhVXJsLnN1YnN0cmluZyhpbmRleCk7XG4gICAgICAgICAganNaaXAuZmlsZShgZmFjZS0ke3Bvc2UudGltZU1pbGlzZWNvbmRzfS4ke2ltYWdlRmlsZUV4dH1gLCBiYXNlNjQsIHtcbiAgICAgICAgICAgIGJhc2U2NDogdHJ1ZSxcbiAgICAgICAgICB9KTtcbiAgICAgICAgfSBjYXRjaCAoZXJyb3IpIHtcbiAgICAgICAgICBjb25zb2xlLndhcm4oXG4gICAgICAgICAgICBgW1Bvc2VFeHBvcnRlclNlcnZpY2VdIHB1c2ggLSBDb3VsZCBub3QgcHVzaCBmYWNlIGZyYW1lIGltYWdlYCxcbiAgICAgICAgICAgIGVycm9yXG4gICAgICAgICAgKTtcbiAgICAgICAgICB0aHJvdyBlcnJvcjtcbiAgICAgICAgfVxuICAgICAgfVxuICAgIH1cblxuICAgIHJldHVybiBhd2FpdCBqc1ppcC5nZW5lcmF0ZUFzeW5jKHsgdHlwZTogJ2Jsb2InIH0pO1xuICB9XG5cbiAgZ2V0RmlsZUV4dGVuc2lvbkJ5TWltZShJTUFHRV9NSU1FOiBzdHJpbmcpIHtcbiAgICBzd2l0Y2ggKElNQUdFX01JTUUpIHtcbiAgICAgIGNhc2UgJ2ltYWdlL3BuZyc6XG4gICAgICAgIHJldHVybiAncG5nJztcbiAgICAgIGNhc2UgJ2ltYWdlL2pwZWcnOlxuICAgICAgICByZXR1cm4gJ2pwZyc7XG4gICAgICBjYXNlICdpbWFnZS93ZWJwJzpcbiAgICAgICAgcmV0dXJuICd3ZWJwJztcbiAgICAgIGRlZmF1bHQ6XG4gICAgICAgIHJldHVybiAncG5nJztcbiAgICB9XG4gIH1cblxuICBwdWJsaWMgYXN5bmMgZ2V0SnNvbigpOiBQcm9taXNlPHN0cmluZz4ge1xuICAgIGlmICh0aGlzLnZpZGVvTWV0YWRhdGEgPT09IHVuZGVmaW5lZCB8fCB0aGlzLnBvc2VzID09PSB1bmRlZmluZWQpXG4gICAgICByZXR1cm4gJ3t9JztcblxuICAgIGlmICghdGhpcy5pc0ZpbmFsaXplZCkge1xuICAgICAgYXdhaXQgdGhpcy5maW5hbGl6ZSgpO1xuICAgIH1cblxuICAgIGxldCBwb3NlTGFuZG1hcmtNYXBwaW5ncyA9IFtdO1xuICAgIGZvciAoY29uc3Qga2V5IG9mIE9iamVjdC5rZXlzKFBPU0VfTEFORE1BUktTKSkge1xuICAgICAgY29uc3QgaW5kZXg6IG51bWJlciA9IFBPU0VfTEFORE1BUktTW2tleSBhcyBrZXlvZiB0eXBlb2YgUE9TRV9MQU5ETUFSS1NdO1xuICAgICAgcG9zZUxhbmRtYXJrTWFwcGluZ3NbaW5kZXhdID0ga2V5O1xuICAgIH1cblxuICAgIGNvbnN0IGpzb246IFBvc2VTZXRKc29uID0ge1xuICAgICAgZ2VuZXJhdG9yOiAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InLFxuICAgICAgdmVyc2lvbjogMSxcbiAgICAgIHZpZGVvOiB0aGlzLnZpZGVvTWV0YWRhdGEhLFxuICAgICAgcG9zZXM6IHRoaXMucG9zZXMubWFwKChwb3NlOiBQb3NlU2V0SXRlbSk6IFBvc2VTZXRKc29uSXRlbSA9PiB7XG4gICAgICAgIC8vIEJvZHlWZWN0b3Ig44Gu5Zyn57iuXG4gICAgICAgIGNvbnN0IGJvZHlWZWN0b3IgPSBbXTtcbiAgICAgICAgZm9yIChjb25zdCBrZXkgb2YgUG9zZVNldC5CT0RZX1ZFQ1RPUl9NQVBQSU5HUykge1xuICAgICAgICAgIGJvZHlWZWN0b3IucHVzaChwb3NlLmJvZHlWZWN0b3Jba2V5IGFzIGtleW9mIEJvZHlWZWN0b3JdKTtcbiAgICAgICAgfVxuXG4gICAgICAgIC8vIEhhbmRWZWN0b3Ig44Gu5Zyn57iuXG4gICAgICAgIGxldCBoYW5kVmVjdG9yOiAobnVtYmVyW10gfCBudWxsKVtdIHwgdW5kZWZpbmVkID0gdW5kZWZpbmVkO1xuICAgICAgICBpZiAocG9zZS5oYW5kVmVjdG9yKSB7XG4gICAgICAgICAgaGFuZFZlY3RvciA9IFtdO1xuICAgICAgICAgIGZvciAoY29uc3Qga2V5IG9mIFBvc2VTZXQuSEFORF9WRUNUT1JfTUFQUElOR1MpIHtcbiAgICAgICAgICAgIGhhbmRWZWN0b3IucHVzaChwb3NlLmhhbmRWZWN0b3Jba2V5IGFzIGtleW9mIEhhbmRWZWN0b3JdKTtcbiAgICAgICAgICB9XG4gICAgICAgIH1cblxuICAgICAgICAvLyBQb3NlU2V0SnNvbkl0ZW0g44GuIHBvc2Ug44Kq44OW44K444Kn44Kv44OI44KS55Sf5oiQXG4gICAgICAgIHJldHVybiB7XG4gICAgICAgICAgdDogcG9zZS50aW1lTWlsaXNlY29uZHMsXG4gICAgICAgICAgZDogcG9zZS5kdXJhdGlvbk1pbGlzZWNvbmRzLFxuICAgICAgICAgIHA6IHBvc2UucG9zZSxcbiAgICAgICAgICBsOiBwb3NlLmxlZnRIYW5kLFxuICAgICAgICAgIHI6IHBvc2UucmlnaHRIYW5kLFxuICAgICAgICAgIHY6IGJvZHlWZWN0b3IsXG4gICAgICAgICAgaDogaGFuZFZlY3RvcixcbiAgICAgICAgICBlOiBwb3NlLmV4dGVuZGVkRGF0YSxcbiAgICAgICAgfTtcbiAgICAgIH0pLFxuICAgICAgcG9zZUxhbmRtYXJrTWFwcHBpbmdzOiBwb3NlTGFuZG1hcmtNYXBwaW5ncyxcbiAgICB9O1xuXG4gICAgcmV0dXJuIEpTT04uc3RyaW5naWZ5KGpzb24pO1xuICB9XG5cbiAgbG9hZEpzb24oanNvbjogc3RyaW5nIHwgYW55KSB7XG4gICAgY29uc3QgcGFyc2VkSnNvbiA9IHR5cGVvZiBqc29uID09PSAnc3RyaW5nJyA/IEpTT04ucGFyc2UoanNvbikgOiBqc29uO1xuXG4gICAgaWYgKHBhcnNlZEpzb24uZ2VuZXJhdG9yICE9PSAnbXAtdmlkZW8tcG9zZS1leHRyYWN0b3InKSB7XG4gICAgICB0aHJvdyAn5LiN5q2j44Gq44OV44Kh44Kk44OrJztcbiAgICB9IGVsc2UgaWYgKHBhcnNlZEpzb24udmVyc2lvbiAhPT0gMSkge1xuICAgICAgdGhyb3cgJ+acquWvvuW/nOOBruODkOODvOOCuOODp+ODsyc7XG4gICAgfVxuXG4gICAgdGhpcy52aWRlb01ldGFkYXRhID0gcGFyc2VkSnNvbi52aWRlbztcbiAgICB0aGlzLnBvc2VzID0gcGFyc2VkSnNvbi5wb3Nlcy5tYXAoKGl0ZW06IFBvc2VTZXRKc29uSXRlbSk6IFBvc2VTZXRJdGVtID0+IHtcbiAgICAgIGNvbnN0IGJvZHlWZWN0b3I6IGFueSA9IHt9O1xuICAgICAgUG9zZVNldC5CT0RZX1ZFQ1RPUl9NQVBQSU5HUy5tYXAoKGtleSwgaW5kZXgpID0+IHtcbiAgICAgICAgYm9keVZlY3RvcltrZXkgYXMga2V5b2YgQm9keVZlY3Rvcl0gPSBpdGVtLnZbaW5kZXhdO1xuICAgICAgfSk7XG5cbiAgICAgIGNvbnN0IGhhbmRWZWN0b3I6IGFueSA9IHt9O1xuICAgICAgaWYgKGl0ZW0uaCkge1xuICAgICAgICBQb3NlU2V0LkhBTkRfVkVDVE9SX01BUFBJTkdTLm1hcCgoa2V5LCBpbmRleCkgPT4ge1xuICAgICAgICAgIGhhbmRWZWN0b3Jba2V5IGFzIGtleW9mIEhhbmRWZWN0b3JdID0gaXRlbS5oIVtpbmRleF07XG4gICAgICAgIH0pO1xuICAgICAgfVxuXG4gICAgICByZXR1cm4ge1xuICAgICAgICB0aW1lTWlsaXNlY29uZHM6IGl0ZW0udCxcbiAgICAgICAgZHVyYXRpb25NaWxpc2Vjb25kczogaXRlbS5kLFxuICAgICAgICBwb3NlOiBpdGVtLnAsXG4gICAgICAgIGxlZnRIYW5kOiBpdGVtLmwsXG4gICAgICAgIHJpZ2h0SGFuZDogaXRlbS5yLFxuICAgICAgICBib2R5VmVjdG9yOiBib2R5VmVjdG9yLFxuICAgICAgICBoYW5kVmVjdG9yOiBoYW5kVmVjdG9yLFxuICAgICAgICBmcmFtZUltYWdlRGF0YVVybDogdW5kZWZpbmVkLFxuICAgICAgICBleHRlbmRlZERhdGE6IGl0ZW0uZSxcbiAgICAgIH07XG4gICAgfSk7XG4gIH1cblxuICBhc3luYyBsb2FkWmlwKGJ1ZmZlcjogQXJyYXlCdWZmZXIsIGluY2x1ZGVJbWFnZXM6IGJvb2xlYW4gPSB0cnVlKSB7XG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBsb2FkWmlwLi4uYCwgSlNaaXApO1xuICAgIGNvbnN0IGpzWmlwID0gbmV3IEpTWmlwKCk7XG4gICAgY29uc29sZS5sb2coYFtQb3NlU2V0XSBpbml0Li4uYCk7XG4gICAgY29uc3QgemlwID0gYXdhaXQganNaaXAubG9hZEFzeW5jKGJ1ZmZlciwgeyBiYXNlNjQ6IGZhbHNlIH0pO1xuICAgIGlmICghemlwKSB0aHJvdyAnWklQ44OV44Kh44Kk44Or44KS6Kqt44G/6L6844KB44G+44Gb44KT44Gn44GX44GfJztcblxuICAgIGNvbnN0IGpzb24gPSBhd2FpdCB6aXAuZmlsZSgncG9zZXMuanNvbicpPy5hc3luYygndGV4dCcpO1xuICAgIGlmIChqc29uID09PSB1bmRlZmluZWQpIHtcbiAgICAgIHRocm93ICdaSVDjg5XjgqHjgqTjg6vjgasgcG9zZS5qc29uIOOBjOWQq+OBvuOCjOOBpuOBhOOBvuOBm+OCkyc7XG4gICAgfVxuXG4gICAgdGhpcy5sb2FkSnNvbihqc29uKTtcblxuICAgIGNvbnN0IGZpbGVFeHQgPSB0aGlzLmdldEZpbGVFeHRlbnNpb25CeU1pbWUodGhpcy5JTUFHRV9NSU1FKTtcblxuICAgIGlmIChpbmNsdWRlSW1hZ2VzKSB7XG4gICAgICBmb3IgKGNvbnN0IHBvc2Ugb2YgdGhpcy5wb3Nlcykge1xuICAgICAgICBpZiAoIXBvc2UuZnJhbWVJbWFnZURhdGFVcmwpIHtcbiAgICAgICAgICBjb25zdCBmcmFtZUltYWdlRmlsZU5hbWUgPSBgZnJhbWUtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKGZyYW1lSW1hZ2VGaWxlTmFtZSlcbiAgICAgICAgICAgID8uYXN5bmMoJ2Jhc2U2NCcpO1xuICAgICAgICAgIGlmIChpbWFnZUJhc2U2NCkge1xuICAgICAgICAgICAgcG9zZS5mcmFtZUltYWdlRGF0YVVybCA9IGBkYXRhOiR7dGhpcy5JTUFHRV9NSU1FfTtiYXNlNjQsJHtpbWFnZUJhc2U2NH1gO1xuICAgICAgICAgIH1cbiAgICAgICAgfVxuICAgICAgICBpZiAoIXBvc2UucG9zZUltYWdlRGF0YVVybCkge1xuICAgICAgICAgIGNvbnN0IHBvc2VJbWFnZUZpbGVOYW1lID0gYHBvc2UtJHtwb3NlLnRpbWVNaWxpc2Vjb25kc30uJHtmaWxlRXh0fWA7XG4gICAgICAgICAgY29uc3QgaW1hZ2VCYXNlNjQgPSBhd2FpdCB6aXBcbiAgICAgICAgICAgIC5maWxlKHBvc2VJbWFnZUZpbGVOYW1lKVxuICAgICAgICAgICAgPy5hc3luYygnYmFzZTY0Jyk7XG4gICAgICAgICAgaWYgKGltYWdlQmFzZTY0KSB7XG4gICAgICAgICAgICBwb3NlLnBvc2VJbWFnZURhdGFVcmwgPSBgZGF0YToke3RoaXMuSU1BR0VfTUlNRX07YmFzZTY0LCR7aW1hZ2VCYXNlNjR9YDtcbiAgICAgICAgICB9XG4gICAgICAgIH1cbiAgICAgIH1cbiAgICB9XG4gIH1cbn1cbiJdfQ==