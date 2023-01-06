import { Injectable } from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseItem, PoseJson, PoseJsonItem, PoseVector } from './pose';

// @ts-ignore
const cosSimilarity = require('cos-similarity');

/**
 * ポーズを管理するためのサービス
 *
 * ※ シングルトンなサービスではないため、Component で providers に指定して使用することを想定
 */
@Injectable()
export class PoseExporterService {
  private videoName?: string;
  private videoMetadata?: {
    width: number;
    height: number;
    duration: number;
  };

  private poses: PoseItem[] = [];
  private isFinalized = false;

  private readonly POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
  ];

  // 全フレームから重複したポーズを削除するかどうか
  private readonly IS_ENABLE_DUPLICATED_POSE_REDUCTION = true;

  constructor(private snackBar: MatSnackBar) {}

  init(videoName: string) {
    this.videoName = videoName;
    this.poses = [];
    this.isFinalized = false;
  }

  finalize() {
    if (this.IS_ENABLE_DUPLICATED_POSE_REDUCTION) {
      // 全ポーズを走査して、類似するポーズを削除する
      const newPoses: PoseItem[] = [];
      for (const poseA of this.poses) {
        let isDuplicated = false;
        for (const poseB of newPoses) {
          if (this.isSimilarPose(poseA.vectors, poseB.vectors)) {
            isDuplicated = true;
            break;
          }
        }
        if (isDuplicated) continue;

        newPoses.push(poseA);
      }

      console.info(
        `[PoseExporterService] getJson - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`
      );
      this.poses = newPoses;
    }

    this.isFinalized = true;
  }

  loadJson(json: string) {
    const parsedJson = JSON.parse(json);

    if (parsedJson.generator !== 'mp-video-pose-extractor') {
      throw '不正なファイル';
    } else if (parsedJson.version !== 1) {
      throw '未対応のバージョン';
    }

    this.videoMetadata = parsedJson.video;
    this.poses = parsedJson.poses.map(
      (poseJsonItem: PoseJsonItem): PoseItem => {
        const poseVector: any = {};
        this.POSE_VECTOR_MAPPINGS.map((key, index) => {
          poseVector[key as keyof PoseVector] = poseJsonItem.vectors[index];
        });

        return {
          t: poseJsonItem.t,
          pose: poseJsonItem.pose,
          vectors: poseVector,
          frameImageDataUrl: undefined,
        };
      }
    );
  }

  async loadZip(buffer: ArrayBuffer, includeImages: boolean = true) {
    const jsZip = new JSZip();
    const zip = await jsZip.loadAsync(buffer, { base64: false });
    if (!zip) throw 'ZIPファイルを読み込めませんでした';

    const json = await zip.file('poses.json')?.async('text');
    if (json === undefined) {
      throw 'ZIPファイルに pose.json が含まれていません';
    }

    this.loadJson(json);

    if (includeImages) {
      for (const pose of this.poses) {
        const frameImageFileName = `snapshot-${pose.t}.jpg`;
        const imageBase64 = await zip.file(frameImageFileName)?.async('base64');
        if (imageBase64 === undefined && !pose.frameImageDataUrl) {
          continue;
        }

        pose.frameImageDataUrl = `data:image/jpeg;base64,${imageBase64}`;
      }
    }
  }

  getNumberOfPoses(): number {
    return this.poses.length;
  }

  getPoses(): PoseItem[] {
    return this.poses;
  }

  pushPose(
    videoTimeMiliseconds: number,
    frameImageJpegDataUrl: string | undefined,
    videoWidth: number,
    videoHeight: number,
    videoDuration: number,
    results: Results
  ) {
    this.videoMetadata = {
      width: videoWidth,
      height: videoHeight,
      duration: videoDuration,
    };

    if (results.poseLandmarks === undefined) return;

    const poseLandmarksWithWorldCoordinate: any[] = (results as any).ea
      ? (results as any).ea
      : [];
    if (poseLandmarksWithWorldCoordinate.length === 0) {
      console.warn(
        `[PoseExporterService] pushPose - Could not get the pose with the world coordinate`,
        results
      );
      return;
    }

    const poseVector = this.getPoseVector(poseLandmarksWithWorldCoordinate);
    if (!poseVector) {
      console.warn(
        `[PoseExporterService] pushPose - Could not get the pose vector`,
        poseLandmarksWithWorldCoordinate
      );
      return;
    }

    const pose: PoseItem = {
      t: videoTimeMiliseconds,
      pose: poseLandmarksWithWorldCoordinate.map((landmark) => {
        return [landmark.x, landmark.y, landmark.z, landmark.visibility];
      }),
      vectors: poseVector,
      frameImageDataUrl: frameImageJpegDataUrl,
    };

    if (1 <= this.poses.length) {
      const lastPose = this.poses[this.poses.length - 1];
      if (this.isSimilarPose(lastPose.vectors, pose.vectors)) {
        return;
      }
    }

    this.poses.push(pose);
  }

  getSimilarPoses(results: Results): PoseItem[] {
    const poseVector = this.getPoseVector((results as any).ea);
    if (!poseVector) throw 'Could not get the pose vector';

    return this.poses.filter((p) => this.isSimilarPose(p.vectors, poseVector));
  }

  private getPoseVector(
    poseLandmarks: { x: number; y: number; z: number }[]
  ): PoseVector | undefined {
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

  downloadAsJson() {
    const message = this.snackBar.open(
      '保存するデータを生成しています... しばらくお待ちください...'
    );

    const blob = new Blob([this.getJson()], {
      type: 'application/json',
    });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${this.videoName}-poses.json`;
    a.click();

    message.dismiss();
  }

  async downloadAsZip() {
    const message = this.snackBar.open(
      '保存するデータを生成しています... しばらくお待ちください...'
    );

    const jsZip = new JSZip();
    jsZip.file('poses.json', this.getJson());

    for (const pose of this.poses) {
      if (!pose.frameImageDataUrl) continue;
      try {
        const index =
          pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
        const base64 = pose.frameImageDataUrl.substring(index);

        jsZip.file(`snapshot-${pose.t}.jpg`, base64, { base64: true });
      } catch (error) {
        console.warn(
          `[PoseExporterService] push - Could not push frame image`,
          error
        );
      }
    }

    const content = await jsZip.generateAsync({ type: 'blob' });
    const url = window.URL.createObjectURL(content);
    const a = document.createElement('a');
    a.href = url;
    a.target = '_blank';
    a.download = `${this.videoName}-poses.zip`;
    a.click();

    message.dismiss();
  }

  private getJson(): string {
    if (this.videoName === undefined) return '{}';

    if (!this.isFinalized) {
      this.finalize();
    }

    let poseLandmarkMappings = [];
    for (const key of Object.keys(POSE_LANDMARKS)) {
      const index: number = POSE_LANDMARKS[key as keyof typeof POSE_LANDMARKS];
      poseLandmarkMappings[index] = key;
    }

    const json: PoseJson = {
      generator: 'mp-video-pose-extractor',
      version: 1,
      video: this.videoMetadata!,
      poses: this.poses.map((pose: PoseItem): PoseJsonItem => {
        const poseVector = [];
        for (const key of this.POSE_VECTOR_MAPPINGS) {
          poseVector.push(pose.vectors[key as keyof PoseVector]);
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

  private isSimilarPose(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector,
    threshold = 0.9
  ): boolean {
    const cosSimilarities = {
      leftWristToLeftElbow: cosSimilarity(
        poseVectorA.leftWristToLeftElbow,
        poseVectorB.leftWristToLeftElbow
      ),
      leftElbowToLeftShoulder: cosSimilarity(
        poseVectorA.leftElbowToLeftShoulder,
        poseVectorB.leftElbowToLeftShoulder
      ),
      rightWristToRightElbow: cosSimilarity(
        poseVectorA.rightWristToRightElbow,
        poseVectorB.rightWristToRightElbow
      ),
      rightElbowToRightShoulder: cosSimilarity(
        poseVectorA.rightElbowToRightShoulder,
        poseVectorB.rightElbowToRightShoulder
      ),
    };

    let isSimilar = false;
    const cosSimilaritiesSum = Object.values(cosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );
    if (cosSimilaritiesSum >= threshold * Object.keys(cosSimilarities).length)
      isSimilar = true;

    console.log(
      `[PoseExporterService] isSimilarPose`,
      isSimilar,
      cosSimilarities
    );

    return isSimilar;
  }
}
