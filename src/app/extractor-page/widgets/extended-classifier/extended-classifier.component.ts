import {
  ChangeDetectorRef,
  Component,
  EventEmitter,
  Input,
  OnDestroy,
  OnInit,
  Output,
} from '@angular/core';
import { Subscription, timer } from 'rxjs';
import { ExtendedClassifier as ExtendedClassifierDefinition } from '../../interfaces/extended-classifier';

@Component({
  selector: 'app-extended-classifier',
  templateUrl: './extended-classifier.component.html',
  styleUrls: ['./extended-classifier.component.scss'],
})
export class ExtendedClassifierComponent implements OnInit, OnDestroy {
  @Input()
  public extendedClassifierDefinitions: ExtendedClassifierDefinition[] = [];

  @Output()
  public extendedClassifierDefinitionsChange = new EventEmitter<
    ExtendedClassifierDefinition[]
  >();

  private DEFAULT_CLASSIFIER_DEFINITIONS: ExtendedClassifierDefinition[] = [
    {
      name: 'faceExp',
      url: 'https://arisucool.github.io/tf-cg-stage-face-expression-classifier/tfjs/',
      target: 'faceFrameImage',
    },
  ];

  private readonly LOCAL_STORAGE_KEY = 'mpVideoPoseExtractorExtClassifiers';

  private saveTimer?: Subscription;

  constructor() {}

  ngOnInit() {
    this.load();
  }

  ngOnDestroy() {
    if (this.saveTimer) {
      this.saveTimer.unsubscribe();
    }
  }

  addClassifier() {
    this.extendedClassifierDefinitions.push({
      name: '',
      url: '',
      target: 'faceFrameImage', // TODO: 選択できるようにする
    });
  }

  deleteClassifier(index: number) {
    this.extendedClassifierDefinitions.splice(index, 1);
  }

  load() {
    const savedClassifiers = window.localStorage.getItem(
      this.LOCAL_STORAGE_KEY
    );
    if (savedClassifiers) {
      this.extendedClassifierDefinitions = JSON.parse(savedClassifiers);
    } else {
      this.extendedClassifierDefinitions = this.DEFAULT_CLASSIFIER_DEFINITIONS;
    }
    this.extendedClassifierDefinitionsChange.emit(
      this.extendedClassifierDefinitions
    );
  }

  save() {
    if (this.saveTimer) {
      this.saveTimer.unsubscribe();
    }

    this.extendedClassifierDefinitionsChange.emit(
      this.extendedClassifierDefinitions
    );

    this.saveTimer = timer(500).subscribe(() => {
      const classifiers = [];
      for (const classifier of this.extendedClassifierDefinitions) {
        if (classifier.name && classifier.url) {
          classifiers.push(classifier);
        }
      }

      let hasChanged = false;
      if (classifiers.length !== this.DEFAULT_CLASSIFIER_DEFINITIONS.length) {
        hasChanged = true;
      } else {
        for (let i = 0; i < classifiers.length; i++) {
          if (
            classifiers[i].name !==
              this.DEFAULT_CLASSIFIER_DEFINITIONS[i].name ||
            classifiers[i].url !== this.DEFAULT_CLASSIFIER_DEFINITIONS[i].url
          ) {
            hasChanged = true;
            break;
          }
        }
      }

      if (!hasChanged) {
        window.localStorage.removeItem(this.LOCAL_STORAGE_KEY);
        return;
      }

      window.localStorage.setItem(
        this.LOCAL_STORAGE_KEY,
        JSON.stringify(this.extendedClassifierDefinitions)
      );
    });
  }
}
