import { TestBed } from '@angular/core/testing';

import { ExtendedClassifierService } from './extended-classifier.service';

describe('ExtendedClassifierService', () => {
  let service: ExtendedClassifierService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(ExtendedClassifierService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
