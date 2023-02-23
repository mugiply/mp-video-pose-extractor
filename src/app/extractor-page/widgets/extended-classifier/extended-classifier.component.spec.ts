import { ComponentFixture, TestBed } from '@angular/core/testing';

import { ExtendedClassifierComponent } from './extended-classifier.component';

describe('ExtendedClassifierComponent', () => {
  let component: ExtendedClassifierComponent;
  let fixture: ComponentFixture<ExtendedClassifierComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ ExtendedClassifierComponent ]
    })
    .compileComponents();

    fixture = TestBed.createComponent(ExtendedClassifierComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
