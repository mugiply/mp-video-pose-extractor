import { TestBed } from '@angular/core/testing';

import { PoseComposerService } from './pose-composer.service';

describe('PoseComposerService', () => {
  let service: PoseComposerService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PoseComposerService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
