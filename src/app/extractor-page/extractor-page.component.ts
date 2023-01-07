import {
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { GpuBuffer, Results } from '@mediapipe/holistic';
import { PoseComposerService } from 'projects/ngx-mp-pose-extractor/src/lib/services/pose-composer.service';
import {
  Pose,
  PoseExtractorService,
} from 'projects/ngx-mp-pose-extractor/src/public-api';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-extractor-page',
  templateUrl: './extractor-page.component.html',
  styleUrls: ['../shared/shared.scss', './extractor-page.component.scss'],
  providers: [PoseExtractorService, PoseComposerService],
})
export class ExtractorPageComponent implements OnInit, OnDestroy {
  @ViewChild('sourceVideo')
  public sourceVideoElement?: ElementRef;

  public sourceVideoUrl?: SafeResourceUrl;
  public sourceVideoFileName?: string = undefined;

  public posePreviewMediaStream?: MediaStream;
  public handPreviewMediaStream?: MediaStream;

  public state: 'initial' | 'processing' | 'completed' = 'initial';

  public mathFloor = Math.floor;

  public pose?: Pose;

  private onResultsEventEmitterSubscription!: Subscription;

  constructor(
    public poseComposerService: PoseComposerService,
    private poseExtractorService: PoseExtractorService,
    private domSanitizer: DomSanitizer,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.onResultsEventEmitterSubscription =
      this.poseExtractorService.onResultsEventEmitter.subscribe(
        (results: { mpResults: Results; posePreviewImageDataUrl: string }) => {
          this.onPoseDetected(
            results.mpResults,
            results.posePreviewImageDataUrl
          );
        }
      );
  }

  ngOnDestroy(): void {
    if (this.onResultsEventEmitterSubscription) {
      this.onResultsEventEmitterSubscription.unsubscribe();
    }
  }

  onSourceVideoEnded(event: any) {
    if (!this.pose) return;
    this.pose.finalize();

    this.state = 'completed';
    const message = this.snackBar.open('検出が完了しました', '保存');
    message.onAction().subscribe(() => {
      this.downloadPosesAsZip();
    });
  }

  async onChooseSourceVideoFile(event: any) {
    const files: File[] = event.target.files;
    if (files.length === 0) return;

    const videoFile = files[0];
    const videoFileUrl = URL.createObjectURL(videoFile);
    this.sourceVideoUrl =
      this.domSanitizer.bypassSecurityTrustResourceUrl(videoFileUrl);

    this.sourceVideoFileName = videoFile.name;
    const videoName = videoFile.name.split('.').slice(0, -1).join('.');

    this.state = 'processing';
    this.pose = this.poseComposerService.init(videoName);

    await this.onVideoFrame();

    this.posePreviewMediaStream =
      this.poseExtractorService.getPosePreviewMediaStream();

    this.handPreviewMediaStream =
      this.poseExtractorService.getHandPreviewMediaStream();
  }

  async onVideoFrame() {
    const videoElement = this.sourceVideoElement?.nativeElement;
    if (!videoElement) return;

    if (videoElement.paused || videoElement.ended) {
      setTimeout(() => {
        this.onVideoFrame();
      }, 500);
      return;
    }

    await this.poseExtractorService.onVideoFrame(videoElement);
    await new Promise(requestAnimationFrame);
    this.onVideoFrame();
  }

  async onPoseDetected(results: Results, posePreviewImageDataUrl: string) {
    if (!this.pose) return;

    const videoElement = this.sourceVideoElement?.nativeElement;
    if (!videoElement) return;

    const sourceVideoTimeMiliseconds = Math.floor(
      this.sourceVideoElement?.nativeElement.currentTime * 1000
    );

    const sourceVideoDurationMiliseconds = Math.floor(
      this.sourceVideoElement?.nativeElement.duration * 1000
    );

    this.pose.pushPose(
      sourceVideoTimeMiliseconds,
      posePreviewImageDataUrl,
      videoElement.videoWidth,
      videoElement.videoHeight,
      sourceVideoDurationMiliseconds,
      results
    );
  }

  public downloadPosesAsZip() {
    if (!this.pose) return;

    const message = this.snackBar.open('保存するデータを生成しています...');
    try {
      this.poseComposerService.downloadAsZip(this.pose);
    } catch (e: any) {
      console.error(e);
      message.dismiss();
      this.snackBar.open('エラー: ' + e.toString(), 'OK');
      return;
    }

    message.dismiss();
  }

  public downloadPosesAsJson() {
    if (!this.pose) return;

    const message = this.snackBar.open('保存するデータを生成しています...');
    try {
      this.poseComposerService.downloadAsJson(this.pose);
    } catch (e: any) {
      console.error(e);
      message.dismiss();
      this.snackBar.open('エラー: ' + e.toString(), 'OK');
      return;
    }

    message.dismiss();
  }
}
