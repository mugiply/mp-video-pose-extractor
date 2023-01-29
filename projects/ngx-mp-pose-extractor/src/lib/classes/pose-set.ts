import { POSE_LANDMARKS, Results } from '@mediapipe/holistic';
import * as JSZip from 'jszip';
import { PoseSetItem } from '../interfaces/pose-set-item';
import { PoseSetJson } from '../interfaces/pose-set-json';
import { PoseSetJsonItem } from '../interfaces/pose-set-json-item';
import { PoseVector } from '../interfaces/pose-vector';

// @ts-ignore
import cosSimilarity from 'cos-similarity';
import { SimilarPoseItem } from '../interfaces/matched-pose-item';
import { ImageTrimmer } from './internals/image-trimmer';

export class PoseSet {
  public generator?: string;
  public version?: number;
  private videoMetadata!: {
    name: string;
    width: number;
    height: number;
    duration: number;
    firstPoseDetectedTime: number;
  };
  public poses: PoseSetItem[] = [];
  public isFinalized?: boolean = false;

  public static readonly POSE_VECTOR_MAPPINGS = [
    'rightWristToRightElbow',
    'rightElbowToRightShoulder',
    'leftWristToLeftElbow',
    'leftElbowToLeftShoulder',
  ];

  // 画像書き出し時の設定
  private readonly IMAGE_WIDTH: number = 1080;
  private readonly IMAGE_MIME: 'image/jpeg' | 'image/png' | 'image/webp' =
    'image/webp';
  private readonly IMAGE_QUALITY = 0.8;

  // 画像の余白除去
  private readonly IMAGE_MARGIN_TRIMMING_COLOR = '#000000';
  private readonly IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD = 50;

  // 画像の背景色置換
  private readonly IMAGE_BACKGROUND_REPLACE_SRC_COLOR = '#016AFD';
  private readonly IMAGE_BACKGROUND_REPLACE_DST_COLOR = '#FFFFFF00';
  private readonly IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD = 130;

  constructor() {
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

  setVideoName(videoName: string) {
    this.videoMetadata.name = videoName;
  }

  setVideoMetaData(width: number, height: number, duration: number) {
    this.videoMetadata.width = width;
    this.videoMetadata.height = height;
    this.videoMetadata.duration = duration;
  }

  getNumberOfPoses(): number {
    if (this.poses === undefined) return -1;
    return this.poses.length;
  }

  getPoses(): PoseSetItem[] {
    if (this.poses === undefined) return [];
    return this.poses;
  }

  getPoseByTime(timeMiliseconds: number): PoseSetItem | undefined {
    if (this.poses === undefined) return undefined;
    return this.poses.find((pose) => pose.timeMiliseconds === timeMiliseconds);
  }

  pushPose(
    videoTimeMiliseconds: number,
    frameImageDataUrl: string | undefined,
    poseImageDataUrl: string | undefined,
    videoWidth: number,
    videoHeight: number,
    videoDuration: number,
    results: Results
  ) {
    this.setVideoMetaData(videoWidth, videoHeight, videoDuration);

    if (results.poseLandmarks === undefined) return;

    if (this.poses.length === 0) {
      this.videoMetadata.firstPoseDetectedTime = videoTimeMiliseconds;
    }

    const poseLandmarksWithWorldCoordinate: any[] = (results as any).ea
      ? (results as any).ea
      : [];
    if (poseLandmarksWithWorldCoordinate.length === 0) {
      console.warn(
        `[PoseSet] pushPose - Could not get the pose with the world coordinate`,
        results
      );
      return;
    }

    const poseVector = PoseSet.getPoseVector(poseLandmarksWithWorldCoordinate);
    if (!poseVector) {
      console.warn(
        `[PoseSet] pushPose - Could not get the pose vector`,
        poseLandmarksWithWorldCoordinate
      );
      return;
    }

    const pose: PoseSetItem = {
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
      const poseDurationMiliseconds =
        videoTimeMiliseconds - lastPose.timeMiliseconds;
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
        const poseDurationMiliseconds =
          this.videoMetadata.duration - lastPose.timeMiliseconds;
        this.poses[this.poses.length - 1].durationMiliseconds =
          poseDurationMiliseconds;
      }
    }

    // 重複ポーズを除去
    this.removeDuplicatedPoses();

    // 画像のマージンを取得
    console.log(`[PoseSet] finalize - Detecting image margins...`);
    let imageTrimming:
      | {
          marginTop: number;
          marginBottom: number;
          heightNew: number;
          heightOld: number;
          width: number;
        }
      | undefined = undefined;
    for (const pose of this.poses) {
      let imageTrimmer = new ImageTrimmer();
      if (!pose.frameImageDataUrl) {
        continue;
      }
      await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);

      const marginColor = await imageTrimmer.getMarginColor();
      console.log(
        `[PoseSet] finalize - Detected margin color...`,
        pose.timeMiliseconds,
        marginColor
      );
      if (marginColor === null) continue;
      if (marginColor !== this.IMAGE_MARGIN_TRIMMING_COLOR) {
        continue;
      }
      const trimmed = await imageTrimmer.trimMargin(
        marginColor,
        this.IMAGE_MARGIN_TRIMMING_DIFF_THRESHOLD
      );
      if (!trimmed) continue;
      imageTrimming = trimmed;
      console.log(
        `[PoseSet] finalize - Determined image trimming positions...`,
        trimmed
      );
      break;
    }

    // 画像を整形
    for (const pose of this.poses) {
      let imageTrimmer = new ImageTrimmer();
      if (!pose.frameImageDataUrl || !pose.poseImageDataUrl) {
        continue;
      }

      console.log(
        `[PoseSet] finalize - Processing image...`,
        pose.timeMiliseconds
      );

      // 画像を整形 - フレーム画像
      await imageTrimmer.loadByDataUrl(pose.frameImageDataUrl);

      if (imageTrimming) {
        await imageTrimmer.crop(
          0,
          imageTrimming.marginTop,
          imageTrimming.width,
          imageTrimming.heightNew
        );
      }

      await imageTrimmer.replaceColor(
        this.IMAGE_BACKGROUND_REPLACE_SRC_COLOR,
        this.IMAGE_BACKGROUND_REPLACE_DST_COLOR,
        this.IMAGE_BACKGROUND_REPLACE_DIFF_THRESHOLD
      );

      await imageTrimmer.resizeWithFit({
        width: this.IMAGE_WIDTH,
      });

      let newDataUrl = await imageTrimmer.getDataUrl(
        this.IMAGE_MIME,
        this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
          ? this.IMAGE_QUALITY
          : undefined
      );
      if (!newDataUrl) {
        console.warn(
          `[PoseSet] finalize - Could not get the new dataurl for frame image`
        );
        continue;
      }
      pose.frameImageDataUrl = newDataUrl;

      // 画像を整形 - ポーズプレビュー画像
      imageTrimmer = new ImageTrimmer();
      await imageTrimmer.loadByDataUrl(pose.poseImageDataUrl);

      if (imageTrimming) {
        await imageTrimmer.crop(
          0,
          imageTrimming.marginTop,
          imageTrimming.width,
          imageTrimming.heightNew
        );
      }

      await imageTrimmer.resizeWithFit({
        width: this.IMAGE_WIDTH,
      });

      newDataUrl = await imageTrimmer.getDataUrl(
        this.IMAGE_MIME,
        this.IMAGE_MIME === 'image/jpeg' || this.IMAGE_MIME === 'image/webp'
          ? this.IMAGE_QUALITY
          : undefined
      );
      if (!newDataUrl) {
        console.warn(
          `[PoseSet] finalize - Could not get the new dataurl for pose preview image`
        );
        continue;
      }
      pose.poseImageDataUrl = newDataUrl;
    }

    this.isFinalized = true;
  }

  removeDuplicatedPoses(): void {
    // 全ポーズを比較して類似ポーズを削除
    const newPoses: PoseSetItem[] = [];
    for (const poseA of this.poses) {
      let isDuplicated = false;
      for (const poseB of newPoses) {
        if (PoseSet.isSimilarPose(poseA.vectors, poseB.vectors)) {
          isDuplicated = true;
          break;
        }
      }
      if (isDuplicated) continue;

      newPoses.push(poseA);
    }

    console.info(
      `[PoseSet] removeDuplicatedPoses - Reduced ${this.poses.length} poses -> ${newPoses.length} poses`
    );
    this.poses = newPoses;
  }

  getSimilarPoses(
    results: Results,
    threshold: number = 0.9
  ): SimilarPoseItem[] {
    const poseVector = PoseSet.getPoseVector((results as any).ea);
    if (!poseVector) throw 'Could not get the pose vector';

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

  static getPoseVector(
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

  static isSimilarPose(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector,
    threshold = 0.9
  ): boolean {
    let isSimilar = false;
    const similarity = PoseSet.getPoseSimilarity(poseVectorA, poseVectorB);
    if (similarity >= threshold) isSimilar = true;

    // console.log(`[PoseSet] isSimilarPose`, isSimilar, similarity);

    return isSimilar;
  }

  static getPoseSimilarity(
    poseVectorA: PoseVector,
    poseVectorB: PoseVector
  ): number {
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

    const cosSimilaritiesSum = Object.values(cosSimilarities).reduce(
      (sum, value) => sum + value,
      0
    );
    return cosSimilaritiesSum / Object.keys(cosSimilarities).length;
  }

  public async getZip(): Promise<Blob> {
    const jsZip = new JSZip();
    jsZip.file('poses.json', await this.getJson());

    const imageFileExt = this.getFileExtensionByMime(this.IMAGE_MIME);

    for (const pose of this.poses) {
      if (pose.frameImageDataUrl) {
        try {
          const index =
            pose.frameImageDataUrl.indexOf('base64,') + 'base64,'.length;
          const base64 = pose.frameImageDataUrl.substring(index);
          jsZip.file(`frame-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
            base64: true,
          });
        } catch (error) {
          console.warn(
            `[PoseExporterService] push - Could not push frame image`,
            error
          );
          throw error;
        }
      }
      if (pose.poseImageDataUrl) {
        try {
          const index =
            pose.poseImageDataUrl.indexOf('base64,') + 'base64,'.length;
          const base64 = pose.poseImageDataUrl.substring(index);
          jsZip.file(`pose-${pose.timeMiliseconds}.${imageFileExt}`, base64, {
            base64: true,
          });
        } catch (error) {
          console.warn(
            `[PoseExporterService] push - Could not push frame image`,
            error
          );
          throw error;
        }
      }
    }

    return await jsZip.generateAsync({ type: 'blob' });
  }

  getFileExtensionByMime(IMAGE_MIME: string) {
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

  public async getJson(): Promise<string> {
    if (this.videoMetadata === undefined || this.poses === undefined)
      return '{}';

    if (!this.isFinalized) {
      await this.finalize();
    }

    let poseLandmarkMappings = [];
    for (const key of Object.keys(POSE_LANDMARKS)) {
      const index: number = POSE_LANDMARKS[key as keyof typeof POSE_LANDMARKS];
      poseLandmarkMappings[index] = key;
    }

    const json: PoseSetJson = {
      generator: 'mp-video-pose-extractor',
      version: 1,
      video: this.videoMetadata!,
      poses: this.poses.map((pose: PoseSetItem): PoseSetJsonItem => {
        const poseVector = [];
        for (const key of PoseSet.POSE_VECTOR_MAPPINGS) {
          poseVector.push(pose.vectors[key as keyof PoseVector]);
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

  loadJson(json: string | any) {
    const parsedJson = typeof json === 'string' ? JSON.parse(json) : json;

    if (parsedJson.generator !== 'mp-video-pose-extractor') {
      throw '不正なファイル';
    } else if (parsedJson.version !== 1) {
      throw '未対応のバージョン';
    }

    this.videoMetadata = parsedJson.video;
    this.poses = parsedJson.poses.map((item: PoseSetJsonItem): PoseSetItem => {
      const poseVector: any = {};
      PoseSet.POSE_VECTOR_MAPPINGS.map((key, index) => {
        poseVector[key as keyof PoseVector] = item.v[index];
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

  async loadZip(buffer: ArrayBuffer, includeImages: boolean = true) {
    console.log(`[PoseSet] loadZip...`, JSZip);
    const jsZip = new JSZip();
    console.log(`[PoseSet] init...`);
    const zip = await jsZip.loadAsync(buffer, { base64: false });
    if (!zip) throw 'ZIPファイルを読み込めませんでした';

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
