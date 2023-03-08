import {
  ChangeDetectorRef,
  Component,
  ElementRef,
  NgZone,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { PoseComposerService } from 'projects/ngx-mp-pose-extractor/src/lib/services/pose-composer.service';
import {
  PoseSet,
  PoseExtractorService,
  OnResultsEvent,
} from 'projects/ngx-mp-pose-extractor/src/public-api';
import { Subscription } from 'rxjs';
import { ExtendedClassifier } from './interfaces/extended-classifier';

// @ts-ignore
import { MP4Demuxer } from './mp4-decorder/demuxer_mp4.mjs';
import { ExtendedClassifierService } from './services/extended-classifier.service';

@Component({
  selector: 'app-extractor-page',
  templateUrl: './extractor-page.component.html',
  styleUrls: ['../shared/shared.scss', './extractor-page.component.scss'],
  providers: [PoseExtractorService, PoseComposerService],
})
export class ExtractorPageComponent implements OnInit, OnDestroy {
  @ViewChild('sourceVideo')
  public sourceVideoElement!: ElementRef;
  public sourceVideoStream?: MediaStream;
  public sourceVideoFileName?: string = undefined;

  public posePreviewMediaStream?: MediaStream;
  public handPreviewMediaStream?: MediaStream;
  public facePreviewMediaStream?: MediaStream;

  public state: 'initial' | 'loading' | 'processing' | 'completed' = 'initial';

  public mathFloor = Math.floor;

  // プレビューで表示される画像
  public previewImage: 'frame' | 'pose' = 'frame';

  // 現在作成している PoseSet
  public poseSet?: PoseSet;

  // 映像の長さ
  public sourceVideoDuration?: number;

  // 映像の抽出されたフレーム
  public sourceVideoFrames?: {
    dataUrl: string;
    timestamp: number;
    width: number;
    height: number;
  }[];
  public numOfSourceVideoFrames = 0;
  public currentSourceVideoFrame?: {
    dataUrl: string;
    timestamp: number;
    width: number;
    height: number;
  };

  // 映像の抽出完了まで待機するためのタイマー
  private sourceVideoLoadTimer: any = null;

  // 映像のフレームを抽出するためのキャンバスおよび変数
  private readonly MINIMUM_FRAME_INTERVAL_MILISECONDS = 200;
  private sourceFrameCanvas?: HTMLCanvasElement;
  private sourceFrameCanvasContext?: CanvasRenderingContext2D;
  private lastFrameDrawedAt?: number;
  private lastFrameChoosedAt?: number;

  // ポーズ検出の結果を取得するためのサブスクリプション
  private onResultsEventEmitterSubscription!: Subscription;

  // 追加の分類器の設定
  extendedClassifierDefinitions: ExtendedClassifier[] = [];

  constructor(
    public poseComposerService: PoseComposerService,
    private poseExtractorService: PoseExtractorService,
    private extendedClassifierService: ExtendedClassifierService,
    private domSanitizer: DomSanitizer,
    private snackBar: MatSnackBar,
    private ngZone: NgZone,
    private changeDetectorRef: ChangeDetectorRef
  ) {}

  ngOnInit(): void {
    this.onResultsEventEmitterSubscription =
      this.poseExtractorService.onResultsEventEmitter.subscribe(
        (results: OnResultsEvent) => {
          this.onPoseDetected(results);
        }
      );

    // 子コンポーネントにより値が変更されるのを検知
    // (ExtendedClassifierComponetn からの値の変更)
    this.changeDetectorRef.detectChanges();
  }

  ngOnDestroy(): void {
    if (this.onResultsEventEmitterSubscription) {
      this.onResultsEventEmitterSubscription.unsubscribe();
    }
  }

  /**
   * ソース動画が選択されたときの処理
   */
  async onChooseSourceVideoFile(event: any) {
    const files: File[] = event.target.files;
    if (files.length === 0) return;

    const videoFile = files[0];
    const videoFileUrl = URL.createObjectURL(videoFile);

    // 初期化
    this.sourceVideoFileName = videoFile.name;
    const videoName = videoFile.name.split('.').slice(0, -1).join('.');
    this.state = 'loading';
    this.poseSet = this.poseComposerService.init(videoName);

    // 追加の分類器を初期化
    try {
      await this.extendedClassifierService.init(
        this.extendedClassifierDefinitions
      );
    } catch (e: any) {
      this.snackBar.open(
        'エラー: 分類器の初期化に失敗...' + e.message,
        undefined,
        {
          duration: 3000,
        }
      );
    }

    // 動画のデコードを開始
    this.sourceFrameCanvas = document.createElement('canvas');
    this.sourceFrameCanvasContext = this.sourceFrameCanvas.getContext('2d')!;
    this.sourceVideoFrames = [];

    // @ts-ignore
    const decoder = new VideoDecoder({
      output: async (frame: any) => {
        // 動画のフレームを取得したとき
        this.lastFrameDrawedAt = Date.now();
        await this.onVideoFrame(frame);
      },
      error: (e: any) => {
        console.error(
          `[ExtractorPageComponent] onChooseSourceVideoFile - Error occurred`,
          e
        );
      },
    });

    const demuxer = new MP4Demuxer(videoFileUrl, {
      onConfig: (config: {
        codec: string;
        codedWidth: number;
        codedHeight: number;
        description: any;
        track: any;
      }) => {
        this.sourceVideoDuration =
          config.track.movie_duration && config.track.movie_timescale
            ? Math.floor(
                (config.track.movie_duration / config.track.movie_timescale) *
                  1000
              )
            : -1;
        decoder.configure(config);
      },
      onChunk: (chunk: any) => {
        decoder.decode(chunk);
      },
      setStatus: (type: string, message: string) => {
        console.debug(
          `[ExtractorPageComponent] - MP4Demuxer setStatus`,
          type,
          message
        );
      },
    });

    // 動画の全フレームが揃うまで待つ
    this.lastFrameDrawedAt = Date.now();
    this.sourceVideoLoadTimer = setInterval(() => {
      if (Date.now() - this.lastFrameDrawedAt! > 1000) {
        clearInterval(this.sourceVideoLoadTimer);
        this.onVideoAllFramesLoaded();
      }
    }, 1000);
  }

  /**
   * ソース動画のフレームを抽出したときの処理
   * @param frame VideoFrame
   */
  async onVideoFrame(frame: any) {
    const timestampMiliseconds = frame.timestamp / 1000;
    if (
      this.lastFrameChoosedAt !== undefined &&
      timestampMiliseconds - this.lastFrameChoosedAt <
        this.MINIMUM_FRAME_INTERVAL_MILISECONDS
    ) {
      frame.close();
      return;
    }

    this.lastFrameChoosedAt = timestampMiliseconds;
    await this.renderVideoFrame(frame);

    const dataUrl = this.sourceFrameCanvas?.toDataURL();
    if (dataUrl && this.sourceVideoFrames) {
      this.sourceVideoFrames.push({
        dataUrl: dataUrl,
        timestamp: timestampMiliseconds,
        width: frame.codedWidth,
        height: frame.codedHeight,
      });
    }

    frame.close();
  }

  /**
   * ソース動画のフレーム描画
   */
  async renderVideoFrame(frame: any) {
    if (!this.sourceFrameCanvas || !this.sourceFrameCanvasContext) {
      return;
    }

    this.sourceFrameCanvas.width = frame.displayWidth;
    this.sourceFrameCanvas.height = frame.displayHeight;
    this.sourceFrameCanvasContext.drawImage(
      frame,
      0,
      0,
      this.sourceFrameCanvas.width,
      this.sourceFrameCanvas.height
    );
  }

  /**
   * ソース動画の全てのフレームから抽出完了したときの処理
   */
  async onVideoAllFramesLoaded() {
    if (!this.sourceVideoFrames || !this.poseSet) {
      return;
    }

    this.numOfSourceVideoFrames = this.sourceVideoFrames.length;

    if (this.numOfSourceVideoFrames <= 1) {
      this.snackBar.open('エラー: 動画のフレーム数が少なすぎます');
      this.state = 'initial';
      this.poseSet = undefined;
      return;
    }

    const message = this.snackBar.open(
      this.numOfSourceVideoFrames +
        ' frames からポーズ検出をおこなっています... '
    );
    this.state = 'processing';
    console.debug(
      `[ExtractorPageComponent] - onVideoAllFramesLoaded`,
      this.sourceVideoFrames
    );

    if (!this.sourceVideoStream && this.sourceFrameCanvas) {
      this.sourceVideoStream = this.sourceFrameCanvas.captureStream(30);
    }

    // 動画のメタデータを設定
    this.poseSet.setVideoMetaData(
      this.sourceVideoFrames[0].width,
      this.sourceVideoFrames[0].height,
      this.sourceVideoFrames[this.sourceVideoFrames.length - 1].timestamp
    );

    // 検出されたポーズをプレビューするためのストリームを生成
    this.posePreviewMediaStream =
      this.poseExtractorService.getPosePreviewMediaStream();

    // 検出された手をプレビューするためのストリームを生成
    this.handPreviewMediaStream =
      this.poseExtractorService.getHandPreviewMediaStream();

    // 検出された顔をプレビューするためのストリームを生成
    this.facePreviewMediaStream =
      this.poseExtractorService.getFacePreviewMediaStream();

    // ポーズ検出を開始
    await this.detectPoseOfNextVideoFrame();
  }

  async detectPoseOfNextVideoFrame() {
    if (!this.sourceFrameCanvas || !this.sourceFrameCanvasContext) {
      throw new Error('sourceFrameCanvas is not initialized');
    } else if (!this.sourceVideoFrames) {
      return;
    } else if (this.sourceVideoFrames.length === 0) {
      this.onPoseDetectionCompleted();
      return;
    }

    const frame = this.sourceVideoFrames.shift();
    if (!frame) {
      throw new Error('frame is not initialized');
    }
    this.currentSourceVideoFrame = frame;

    try {
      let img = new Image();
      img.src = frame.dataUrl;
      await img.decode();

      this.sourceFrameCanvas.width = frame.width;
      this.sourceFrameCanvas.height = frame.height;
      this.sourceFrameCanvasContext.drawImage(
        img,
        0,
        0,
        this.sourceFrameCanvas.width,
        this.sourceFrameCanvas.height
      );
    } catch (e) {
      console.error(
        `[ExtractorPageComponent] detectPoseOfNextVideoFrame - Error occurred`,
        e,
        frame
      );
      await this.detectPoseOfNextVideoFrame();
      return;
    }

    this.poseExtractorService.onVideoFrame(this.sourceFrameCanvas);
  }

  /**
   * フレームからポーズを検出したときの処理
   */
  async onPoseDetected(results: OnResultsEvent) {
    if (!this.poseSet || !this.currentSourceVideoFrame) return;

    // フレームを取得
    const frame = this.currentSourceVideoFrame;

    // フレームの時間を取得
    const sourceVideoTimeMiliseconds = Math.floor(frame.timestamp);

    // ポーズを追加
    const pose = this.poseSet.pushPose(
      sourceVideoTimeMiliseconds,
      results.frameImageDataUrl,
      results.posePreviewImageDataUrl,
      results.faceFrameImageDataUrl,
      results.mpResults
    );

    // 追加の分類を実行
    if (pose) {
      pose.extendedData = await this.extendedClassifierService.classify(pose);
    }

    // 次のフレームのポーズ検出へ
    await this.detectPoseOfNextVideoFrame();
  }

  /**
   * ポーズの検出が完了したときの処理
   */
  async onPoseDetectionCompleted() {
    if (!this.poseSet) return;

    let message = this.snackBar.open('最終処理をしています...');
    await this.poseSet.finalize();
    message.dismiss();

    this.state = 'completed';
    message = this.snackBar.open('検出が完了しました', '保存');
    message.onAction().subscribe(() => {
      this.downloadPosesAsZip();
    });
  }

  /**
   * ポーズセットのダウンロード (ZIP ファイル)
   */
  public downloadPosesAsZip() {
    if (!this.poseSet) return;

    const message = this.snackBar.open('保存するデータを生成しています...');
    try {
      this.poseComposerService.downloadAsZip(this.poseSet);
    } catch (e: any) {
      console.error(e);
      message.dismiss();
      this.snackBar.open('エラー: ' + e.toString(), 'OK');
      return;
    }

    message.dismiss();
  }

  /**
   * ポーズセットのダウンロード (JSON ファイル)
   */
  public downloadPosesAsJson() {
    if (!this.poseSet) return;

    const message = this.snackBar.open('保存するデータを生成しています...');
    try {
      this.poseComposerService.downloadAsJson(this.poseSet);
    } catch (e: any) {
      console.error(e);
      message.dismiss();
      this.snackBar.open('エラー: ' + e.toString(), 'OK');
      return;
    }

    message.dismiss();
  }
}
