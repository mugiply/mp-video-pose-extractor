import {
  Component,
  ElementRef,
  OnDestroy,
  OnInit,
  ViewChild,
} from '@angular/core';
import { MatSnackBar } from '@angular/material/snack-bar';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { Results } from '@mediapipe/holistic';
import { Subscription } from 'rxjs';
import { PoseExporterService } from '../pose-exporter.service';
import { ExtractorService } from './extractor.service';

@Component({
  selector: 'app-extractor',
  templateUrl: './extractor.component.html',
  styleUrls: ['./extractor.component.scss'],
})
export class ExtractorComponent implements OnInit, OnDestroy {
  @ViewChild('sourceVideo')
  public sourceVideoElement?: ElementRef;

  public sourceVideoUrl?: SafeResourceUrl;
  public sourceVideoFileName?: string = undefined;

  public posePreviewMediaStream?: MediaStream;
  public handPreviewMediaStream?: MediaStream;

  public state: 'initial' | 'processing' | 'completed' = 'initial';

  public mathFloor = Math.floor;

  private onResultsEventEmitterSubscription!: Subscription;

  constructor(
    public poseExporterService: PoseExporterService,
    private domSanitizer: DomSanitizer,
    private extractorService: ExtractorService,
    private snackBar: MatSnackBar
  ) {}

  ngOnInit(): void {
    this.onResultsEventEmitterSubscription =
      this.extractorService.onResultsEventEmitter.subscribe((results: any) => {
        this.onPoseDetected(results);
      });
  }

  ngOnDestroy(): void {
    if (this.onResultsEventEmitterSubscription) {
      this.onResultsEventEmitterSubscription.unsubscribe();
    }
  }

  onSourceVideoEnded(event: any) {
    this.state = 'completed';
    const message = this.snackBar.open('検出が完了しました', '保存', {
      duration: 5000,
    });
    message.onAction().subscribe(() => {
      this.poseExporterService.downloadAsJson();
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
    this.poseExporterService.start(videoName);

    await this.onVideoFrame();

    this.posePreviewMediaStream =
      this.extractorService.getPosePreviewMediaStream();

    this.handPreviewMediaStream =
      this.extractorService.getHandPreviewMediaStream();
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

    await this.extractorService.onVideoFrame(videoElement);
    await new Promise(requestAnimationFrame);
    this.onVideoFrame();
  }

  async onPoseDetected(results: Results) {
    const videoElement = this.sourceVideoElement?.nativeElement;
    if (!videoElement) return;

    const sourceVideoTimeMiliseconds = Math.floor(
      this.sourceVideoElement?.nativeElement.currentTime * 1000
    );
    this.poseExporterService.push(
      sourceVideoTimeMiliseconds,
      videoElement.videoWidth,
      videoElement.videoHeight,
      results
    );
  }
}
